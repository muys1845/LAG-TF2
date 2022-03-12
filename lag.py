import glob
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
import layers
import time
import numpy as np
import matplotlib.pyplot as plt
import data
import functools
from absl import app
from absl import flags
import json
import imageio
from networks import Generator, Discriminator

FLAGS = flags.FLAGS


class LAG:
    def __init__(self, filters, filters_min, blocks, noise_dim, upscale_start, upscale_stop, saved_dir):
        def ilog2(x):
            return int(np.ceil(np.log2(x)))

        # init the model
        self.lod_min = ilog2(upscale_start)
        self.lod_max = ilog2(upscale_stop)
        lfilters = [max(filters_min, filters >> stage) for stage in range(self.lod_max + 1)]
        self.gen = Generator(lfilters, blocks, noise_dim)
        self.disc = Discriminator(lfilters, blocks)
        self.noise_dim = noise_dim

        # init the saving directories
        self.saved_dir = saved_dir
        self.history_files_dir = os.path.join(saved_dir, 'history_files')
        self.temp_history_dir = os.path.join(saved_dir, 'temp_history')
        self.temp_results_dir = os.path.join(saved_dir, 'temp_results')
        for dir_ in [self.saved_dir, self.history_files_dir, self.temp_history_dir, self.temp_results_dir]:
            os.makedirs(dir_, exist_ok=True)

        self.ckpt = tf.train.Checkpoint(gen=self.gen, disc=self.disc)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, os.path.join(self.saved_dir, 'checkpoints'),
                                                       max_to_keep=2)

        # init the logging file
        self.logging_path = os.path.join(saved_dir, 'training_logging.json')
        if not os.path.exists(self.logging_path):
            with open(self.logging_path, 'w') as f:
                json.dump({'kimg_cur': 0}, f, indent=2)

        # init the history records:
        self.history_items = ['mse_gen', 'adv_gen', 'adv_disc', 'gp_discF']
        history_record_path = os.path.join(self.history_files_dir, 'items.json')
        if not os.path.exists(history_record_path):
            with open(history_record_path, 'w') as f:
                json.dump(self.history_items, f, indent=2)

        self.config = {'model': dict(filters=filters, filters_min=filters_min, blocks=blocks, noise_dim=noise_dim,
                                     upscale_start=upscale_start, upscale_stop=upscale_stop, saved_dir=saved_dir)}

    def generate_and_save_image(self, image, lod, lod_start, lod_stop, saved_path):
        if image.ndim == 3:
            image = image[tf.newaxis, :, :, :]

        lores = layers.downscale(image, 1 << self.lod_max)
        real = layers.downscale(image, scale=1 << (self.lod_max - lod_stop))
        img_noise_free = self.gen(lores, lod=lod, lod_start=lod_start, lod_stop=lod_stop)
        img_noise_normal = self.gen(lores, add_noise=True, lod=lod, lod_start=lod_start, lod_stop=lod_stop)

        title = ['HR', 'LR', 'SR_NOISE_FREE', 'SR_NOISE_NORMAL']
        for title_ind, img in enumerate([real, lores, img_noise_free, img_noise_normal]):
            img = img * 0.5 + 0.5
            img = tf.clip_by_value(img, 0., 1.)
            plt.subplot(2, 2, title_ind + 1)
            plt.imshow(img[0])
            plt.axis('off')
            plt.title(title[title_ind] + ' - ' + str(img.shape[1:3]))
        plt.savefig(saved_path)
        plt.close()

    def __save_training_data(self, phase_ind, period, history, test_data, kimg_cur, lod, lod_start, lod_stop):
        latest_checkpoint = 'phase_{}_period_{}'.format(phase_ind, period)

        # save history of the latest period
        for label, history_, dir_ in zip(['gen_loss', 'disc_loss', 'gp'], history,
                                         [self.gen_loss_dir, self.disc_loss_dir, self.gp_dir]):
            plt.plot(history_, label=label)
            plt.legend()
            plt.savefig(os.path.join(dir_, '{}.jpg'.format(latest_checkpoint)))
            plt.close()

        # save test results
        test_dir = os.path.join(self.test_dir, latest_checkpoint)
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
        for img_ind, image in enumerate(test_data, start=1):
            self.generate_and_save_image(image, lod, lod_start, lod_stop, os.path.join(test_dir, str(img_ind) + '.jpg'))

        # save checkpoint
        self.ckpt_manager.save()

        # update nimg_cur
        with open(self.logging_path, 'w') as f:
            json.dump({'kimg_cur': kimg_cur}, f, indent=2)

    def train(self, dataset, batch_size, training_kimg, transition_kimg, save_freq_kimg, lr_gen, lr_disc,
              mse_weight, wass_target, ttur):
        def train_step(real, lod, lod_start, lod_stop, training=True):
            def straight_through_round(x, r=127.5 / 4):
                xr = tf.round(x * r) / r
                return tf.stop_gradient(xr - x) + x

            gen = functools.partial(self.gen, lod=lod, lod_start=lod_start, lod_stop=lod_stop)
            disc = functools.partial(self.disc, lod=lod, lod_start=lod_start, lod_stop=lod_stop)
            upscale = functools.partial(layers.upscale, n=1 << (self.lod_max - lod_stop))
            downscale = functools.partial(layers.downscale, n=1 << (self.lod_max - lod_stop))
            with tf.GradientTape(persistent=True) as tape:
                lores = layers.downscale(real, 1 << self.lod_max)
                real = downscale(real)
                if lod_stop != lod_start:
                    real = layers.blend_resolution(layers.remove_details2d(real), real, lod - lod_start)

                fake = gen(lores)
                fake_noise = gen(lores, add_noise=True)
                lores_fake = layers.downscale(upscale(fake), 1 << self.lod_max)
                lores_fake_noise = layers.downscale(upscale(fake_noise), 1 << self.lod_max)
                latent_real = disc(real, straight_through_round(tf.abs(lores - lores)))
                latent_fake = disc(fake, straight_through_round(tf.abs(lores - lores_fake)))
                latent_fake_noise = disc(fake_noise, straight_through_round(tf.abs(lores - lores_fake_noise)))

                # Gradient penalty.
                mix = tf.random.uniform([batch_size, 1, 1, 1], 0., 1.)
                mixed = real + mix * (fake_noise - real)
                mixed = upscale(mixed)
                mixed_round = straight_through_round(tf.abs(lores - layers.downscale(mixed, 1 << self.lod_max)))
                mixdown = downscale(mixed)
                grad = tape.gradient(tf.reduce_sum(tf.reduce_mean(disc(mixdown, mixed_round), 3)), mixed)
                grad_norm = tf.sqrt(tf.reduce_mean(tf.square(grad), axis=[1, 2, 3]) + 1e-8)

                loss_dreal = -tf.reduce_mean(latent_real)
                loss_dfake = tf.reduce_mean(latent_fake_noise)
                loss_gfake = -tf.reduce_mean(latent_fake_noise)
                loss_gmse = mse(latent_real, latent_fake)
                loss_gp = 10 * tf.reduce_mean(tf.square(grad_norm - wass_target)) * wass_target ** -2

                loss_gen = loss_gfake + loss_gmse * mse_weight
                loss_disc = (loss_dreal + loss_dfake + loss_gp) * ttur

            grad_gen = tape.gradient(loss_gen, self.gen.trainable_variables)
            grad_disc = tape.gradient(loss_disc, self.disc.trainable_variables)
            if training:
                optm.apply_gradients(zip(grad_gen, self.gen.trainable_variables))
                optm.apply_gradients(zip(grad_disc, self.disc.trainable_variables))
                return loss_gen, loss_disc, loss_gp

        # save the config
        self.config['training'] = dict(data_name=dataset.data_name, batch_size=batch_size,
                                       training_kimg=training_kimg, transition_kimg=transition_kimg,
                                       save_freq_kimg=save_freq_kimg, lr_gen=lr_gen, lr_disc=lr_disc,
                                       mse_weight=mse_weight, wass_target=wass_target, ttur=ttur)
        with open(os.path.join(self.saved_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
            print('training config saved')

        train_data = dataset.train_data.shuffle(1024, reshuffle_each_iteration=True).batch(
            batch_size).as_numpy_iterator()
        test_data = dataset.test_data
        build_data = next(train_data)

        # make a schedule
        schedule = []
        training_nimg = training_kimg << 10
        transition_nimg = transition_kimg << 10
        total_nimg = 0
        for i in range(self.lod_min, self.lod_max):
            if training_kimg:
                schedule.append(data.TrainPhase(total_nimg, total_nimg + training_nimg, i, i))
                total_nimg += training_nimg
            if transition_kimg:
                schedule.append(data.TrainPhase(total_nimg, total_nimg + transition_nimg, i, i + 1))
                total_nimg += transition_nimg
        if training_kimg:
            schedule.append(data.TrainPhase(total_nimg, total_nimg + training_nimg, self.lod_max, self.lod_max))
            total_nimg += training_nimg

        with open(self.logging_path, 'r') as f:
            log = json.load(f)
        nimg_cur = log['kimg_cur'] << 10
        if nimg_cur and self.ckpt_manager.latest_checkpoint:  # load latest checkpoint
            train_data = train_data.skip(nimg_cur % dataset.train_sample)
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print('latest checkpoint restored')
        else:  # build generator and discriminator
            nimg_cur = 0
            print('training in the first time')

        # building
        optm = keras.optimizers.Adam(lr_gen, 0, 0.99)
        mse = keras.losses.MeanSquaredError()
        train_step(build_data, self.lod_max, self.lod_max, self.lod_max, training=False)

        save_freq_nimg = save_freq_kimg << 10
        for ind, phase in enumerate(schedule, start=1):
            if nimg_cur >= phase.nimg_stop:
                continue
            self.gen.update_trainable_layers(phase.lod_stop)
            self.disc.update_trainable_layers(phase.lod_stop)

            func = tf.function(train_step)

            total_batch = (phase.nimg_stop - phase.nimg_start) // batch_size
            print('training on phase: {}/{} - lod: {} - lod_start: {} - lod_stop: {}'.format(
                ind, len(schedule), phase.lod(nimg_cur), phase.lod_start, phase.lod_stop
            ))

            start_time = time.time()
            while nimg_cur < phase.nimg_stop:
                history = np.zeros([len(self.history_items), (save_freq_nimg + batch_size - 1) // batch_size],
                                   dtype=np.float32)
                batch_start = (nimg_cur - phase.nimg_start + batch_size - 1) // batch_size

                for batch in range(save_freq_nimg // batch_size):
                    hires = next(train_data)
                    history[:, batch] = func(hires, phase.lod(nimg_cur), phase.lod_start, phase.lod_stop)
                    nimg_cur += batch_size
                    print('\rbatch: {}/{}'.format(batch_start + batch + 1, total_batch), end='')

                # save temp results
                self.__save_training_data(ind, (nimg_cur - phase.nimg_start) // save_freq_nimg, history,
                                          test_data.as_numpy_iterator(), nimg_cur >> 10,
                                          phase.lod(nimg_cur), phase.lod_start, phase.lod_stop)

            sec = time.time() - start_time
            print(' - time: %dh %dm %ds' % (sec / 3600, sec % 3600 / 60, sec % 60))

        # generate training gif
        gif_dir = os.path.join(self.test_dir, 'gif')
        if not os.path.exists(gif_dir):
            os.mkdir(gif_dir)
        print('generating training gif')
        for i in range(1, dataset.test_sample + 1):
            file_path = os.path.join(gif_dir, '{}.gif'.format(i))
            if not os.path.exists(file_path):
                filenames = glob.glob(self.test_dir + '/phase_*_period_*/{}.jpg'.format(i))
                self.__generate_gif(file_path, filenames)
            print('\r{}/{}'.format(i, dataset.test_sample), end='')
        print('\ntraining complete')
        print('total training images: {}'.format(total_nimg))

    @staticmethod
    def __generate_gif(file_path, filenames):
        with imageio.get_writer(file_path, mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)


def main(_):
    lag = LAG(
        filters=FLAGS.filters,
        filters_min=FLAGS.filters_min,
        blocks=FLAGS.blocks,
        noise_dim=FLAGS.noise_dim,
        upscale_start=FLAGS.upscale_start,
        upscale_stop=FLAGS.upscale_stop,
        saved_dir=FLAGS.saved_dir
    )
    lag.train(
        dataset=data.Dataset(data_name=FLAGS.dataset),
        batch_size=FLAGS.batch_size,
        training_kimg=FLAGS.training_kimg,
        transition_kimg=FLAGS.transition_kimg,
        save_freq_kimg=FLAGS.save_freq_kimg,
        lr_gen=FLAGS.lr_gen,
        lr_disc=FLAGS.lr_disc,
        mse_weight=FLAGS.mse_weight,
        wass_target=FLAGS.wass_target,
        ttur=FLAGS.ttur
    )


if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity('ERROR')
    if not tf.config.list_physical_devices('GPU'):
        raise SystemError('You need at least 1 GPU.')

    flags.DEFINE_string('dataset', 'celeba', 'Name of the training dataset.')

    flags.DEFINE_integer('filters', 256, 'Filter size of first convolution.')
    flags.DEFINE_integer('filters_min', 64, 'Minimum filter size of convolution.')
    flags.DEFINE_integer('blocks', 8, 'Number of residual layers in residual networks.')
    flags.DEFINE_integer('noise_dim', 64, 'Number of noise dimensions to concat to lores.')
    flags.DEFINE_integer('upscale_start', 2, 'Start upscale factor of the growing model.')
    flags.DEFINE_integer('upscale_stop', 32, 'Stop upscale factor of the growing model.')
    flags.DEFINE_string('saved_dir', 'celeba_x32', 'Directory for the training data.')

    flags.DEFINE_integer('batch_size', 64, 'Number of images per batch.')
    flags.DEFINE_integer('training_kimg', 2048, 'Number of images during between transitions (in kimg).')
    flags.DEFINE_integer('transition_kimg', 2048, 'Number of images during transition (in kimg).')
    flags.DEFINE_integer('save_freq_kimg', 256, 'Number of images before saving (in kimg).')

    flags.DEFINE_float('lr_gen', 0.001, 'Learning rate of the generator.')
    flags.DEFINE_float('lr_disc', 0.001, 'Learning rate of the discriminator (i.e. the critic).')
    flags.DEFINE_float('mse_weight', 10, 'Amount of mean square error loss for G.')
    flags.DEFINE_float('wass_target', 1, 'Wasserstein gradient penalty target value.')

    flags.DEFINE_integer('ttur', 4, 'How much faster D is trained.')

    app.run(main)

# for a quick test:
# python lag.py --saved_dir=test --batch_size=2 --training_kimg=2 --transition_kimg=2 --save_freq_kimg=1
