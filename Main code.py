"""
U-Net 3+ cGAN — 裂纹路径预测 (Gen 2: 几何+能量场 → 裂纹)
适用: Windows 11, RTX 3050 4GB, TensorFlow 2.x
"""
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 0. 环境检查
# ============================================================
print(f"Python: {sys.version.split()[0]}")
print(f"TensorFlow: {tf.__version__}")

# GPU 配置 (TF2 方式)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        info = tf.config.experimental.get_device_details(gpus[0])
        vram_mb = info.get('device_memory_size', 'N/A')
        print(f"GPU: {info.get('device_name', 'Unknown')} | 显存: {vram_mb}")
    except Exception as e:
        print(f"GPU 配置警告: {e}")
else:
    print("未检测到 GPU，将使用 CPU (速度会慢很多)")

# ============================================================
# 1. 路径 (适配 Windows 实际目录结构)
# ============================================================
PATH_train = './Train_images/Train_images/'
PATH_test = './Test_images/Test_images/'
OUT_DIR = './Out_rock'
SAVE_DIR = './SaveModel'

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# 检查数据是否存在
if not os.path.isdir(PATH_train):
    print(f"错误: 训练数据目录不存在: {PATH_train}")
    print("请确认 Train_images 文件夹在当前目录下。")
    sys.exit(1)
if not os.path.isdir(PATH_test):
    print(f"错误: 测试数据目录不存在: {PATH_test}")
    sys.exit(1)

train_files = tf.io.gfile.glob(PATH_train + 'Train_*.jpg')
test_files = tf.io.gfile.glob(PATH_test + 'Test_*.jpg')
print(f"训练图片: {len(train_files)} 张")
print(f"测试图片: {len(test_files)} 张")

if len(train_files) == 0:
    print(f"未在 {PATH_train} 找到 Train_*.jpg 文件，请检查路径。")
    sys.exit(1)

# ============================================================
# 2. 图片加载 (3-panel 训练数据 + 5-panel 测试数据)
# ============================================================
def load_3_panel(image_file):
    """训练数据: Panel1=几何, Panel2=能量场, Panel3=裂纹"""
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    w = tf.shape(image)[1] // 3
    input_image = tf.cast(image[:, :w, :], tf.float32)
    energy_image = tf.cast(image[:, w:2*w, :], tf.float32)
    real_image = tf.cast(image[:, 2*w:, :], tf.float32)
    return input_image, energy_image, real_image

def load_5_panel(image_file):
    """测试数据: Panel1=几何, Panel2=能量场, Panel4=裂纹(标准答案)"""
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    w = tf.shape(image)[1] // 5
    input_image = tf.cast(image[:, :w, :], tf.float32)
    energy_image = tf.cast(image[:, w:2*w, :], tf.float32)
    real_image = tf.cast(image[:, 3*w:4*w, :], tf.float32)
    return input_image, energy_image, real_image

# ============================================================
# 3. 数据预处理
# ============================================================
IMG_WIDTH, IMG_HEIGHT = 256, 256

def resize(input_image, energy_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    energy_image = tf.image.resize(energy_image, [height, width],
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, energy_image, real_image

def random_crop(input_image, energy_image, real_image):
    stacked = tf.stack([input_image, energy_image, real_image], axis=0)
    cropped = tf.image.random_crop(stacked, size=[3, IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped[0], cropped[1], cropped[2]

def random_jitter(input_image, energy_image, real_image):
    input_image, energy_image, real_image = resize(
        input_image, energy_image, real_image, 286, 286)
    input_image, energy_image, real_image = random_crop(
        input_image, energy_image, real_image)
    # tf.cond 替代 Python if，适配 AutoGraph
    def do_flip():
        return (tf.image.flip_left_right(input_image),
                tf.image.flip_left_right(energy_image),
                tf.image.flip_left_right(real_image))
    def no_flip():
        return input_image, energy_image, real_image
    return tf.cond(tf.random.uniform(()) > 0.5, do_flip, no_flip)

def normalize(input_image, energy_image, real_image):
    input_image = (input_image / 127.5) - 1
    energy_image = (energy_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, energy_image, real_image

def load_image_train(image_file):
    i, e, r = load_3_panel(image_file)
    i, e, r = random_jitter(i, e, r)
    i, e, r = normalize(i, e, r)
    return i, e, r

def load_image_test(image_file):
    i, e, r = load_5_panel(image_file)
    i, e, r = resize(i, e, r, IMG_HEIGHT, IMG_WIDTH)
    i, e, r = normalize(i, e, r)
    return i, e, r

# ============================================================
# 4. 数据集 (BATCH_SIZE=1 适配 4GB 显存)
# ============================================================
BATCH_SIZE = 1
BUFFER_SIZE = 120

train_dataset = tf.data.Dataset.list_files(PATH_train + 'Train_*.jpg')
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(PATH_test + 'Test_*.jpg')
test_dataset = test_dataset.map(load_image_test,
                                 num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).repeat()

# ============================================================
# 5. U-Net 3+ 网络组件
# ============================================================
OUTPUT_CHANNELS = 3

def downsample(filters, size, stride, apply_batchnorm=True):
    init = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=stride,
              padding='same', kernel_initializer=init, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, stride, apply_dropout=False):
    init = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=stride,
              padding='same', kernel_initializer=init, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.2))
    result.add(tf.keras.layers.ReLU())
    return result

def downpool(filters, size):
    init = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.MaxPooling2D(pool_size=(size, size)))
    result.add(tf.keras.layers.Conv2D(filters, 3, strides=1, padding='same',
              kernel_initializer=init, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample2(filters, size):
    init = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.UpSampling2D(size=(size, size),
              interpolation='bilinear'))
    result.add(tf.keras.layers.Conv2D(filters, 3, strides=1, padding='same',
              kernel_initializer=init, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def conv(filters):
    init = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters * 5, 3, strides=1, padding='same',
              kernel_initializer=init, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

# ============================================================
# 6. Generator (U-Net 3+, 双输入: 几何+能量场)
# ============================================================
def Generator():
    init = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_WIDTH, 3], name='input_image')
    energy_inp = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_WIDTH, 3], name='energy_image')

    xe1 = tf.keras.layers.Concatenate()([inp, energy_inp])
    xe2 = downsample(32, 4, 2, apply_batchnorm=False)(xe1)
    xe3 = downsample(64, 4, 2)(xe2)
    xe4 = downsample(128, 4, 2)(xe3)
    xe5 = downsample(128, 4, 2)(xe4)
    xe6 = downsample(256, 4, 2)(xe5)
    xe7 = downsample(256, 4, 2)(xe6)
    xe8 = downsample(512, 4, 2)(xe7)
    xe9 = downsample(512, 4, 2)(xe8)

    fiter = 32

    x81 = downpool(fiter, 128)(xe1);      x82 = downpool(fiter, 64)(xe2)
    x83 = downpool(fiter, 32)(xe3);       x84 = downpool(fiter, 16)(xe4)
    x85 = downpool(fiter, 8)(xe5);        x86 = downpool(fiter, 4)(xe6)
    x87 = downpool(fiter, 2)(xe7);        x88 = downsample(fiter, 3, 1)(xe8)
    x89 = upsample(fiter, 4, 2, apply_dropout=True)(xe9)
    xd8 = tf.keras.layers.Concatenate()([x89, x88, x87, x86, x85, x84, x83, x82, x81])
    xd8 = conv(288)(xd8)

    x71 = downpool(fiter, 64)(xe1);       x72 = downpool(fiter, 32)(xe2)
    x73 = downpool(fiter, 16)(xe3);       x74 = downpool(fiter, 8)(xe4)
    x75 = downpool(fiter, 4)(xe5);        x76 = downpool(fiter, 2)(xe6)
    x77 = downsample(fiter, 3, 1)(xe7);   x78 = upsample(fiter, 4, 2, apply_dropout=True)(xd8)
    x79 = upsample2(fiter, 4)(xe9)
    xd7 = tf.keras.layers.Concatenate()([x79, x78, x77, x76, x75, x74, x73, x72, x71])
    xd7 = conv(288)(xd7)

    x61 = downpool(fiter, 32)(xe1);       x62 = downpool(fiter, 16)(xe2)
    x63 = downpool(fiter, 8)(xe3);        x64 = downpool(fiter, 4)(xe4)
    x65 = downpool(fiter, 2)(xe5);        x66 = downsample(fiter, 3, 1)(xe6)
    x67 = upsample(fiter, 4, 2, apply_dropout=True)(xd7)
    x68 = upsample2(fiter, 4)(xd8);       x69 = upsample2(fiter, 8)(xe9)
    xd6 = tf.keras.layers.Concatenate()([x69, x68, x67, x66, x65, x64, x63, x62, x61])
    xd6 = conv(288)(xd6)

    x51 = downpool(fiter, 16)(xe1);       x52 = downpool(fiter, 8)(xe2)
    x53 = downpool(fiter, 4)(xe3);        x54 = downpool(fiter, 2)(xe4)
    x55 = downsample(fiter, 3, 1)(xe5);   x56 = upsample(fiter, 4, 2, apply_dropout=True)(xd6)
    x57 = upsample2(fiter, 4)(xd7);       x58 = upsample2(fiter, 8)(xd8)
    x59 = upsample2(fiter, 16)(xe9)
    xd5 = tf.keras.layers.Concatenate()([x59, x58, x57, x56, x55, x54, x53, x52, x51])
    xd5 = conv(288)(xd5)

    x41 = downpool(fiter, 8)(xe1);        x42 = downpool(fiter, 4)(xe2)
    x43 = downpool(fiter, 2)(xe3);        x44 = downsample(fiter, 3, 1)(xe4)
    x45 = upsample(fiter, 4, 2)(xd5);     x46 = upsample2(fiter, 4)(xd6)
    x47 = upsample2(fiter, 8)(xd7);       x48 = upsample2(fiter, 16)(xd8)
    x49 = upsample2(fiter, 32)(xe9)
    xd4 = tf.keras.layers.Concatenate()([x49, x48, x47, x46, x45, x44, x43, x42, x41])
    xd4 = conv(288)(xd4)

    x31 = downpool(fiter, 4)(xe1);        x32 = downpool(fiter, 2)(xe2)
    x33 = downsample(fiter, 3, 1)(xe3);   x34 = upsample(fiter, 4, 2)(xd4)
    x35 = upsample2(fiter, 4)(xd5);       x36 = upsample2(fiter, 8)(xd6)
    x37 = upsample2(fiter, 16)(xd7);      x38 = upsample2(fiter, 32)(xd8)
    x39 = upsample2(fiter, 64)(xe9)
    xd3 = tf.keras.layers.Concatenate()([x39, x38, x37, x36, x35, x34, x33, x32, x31])
    xd3 = conv(288)(xd3)

    x21 = downpool(fiter, 2)(xe1);        x22 = downsample(fiter, 3, 1)(xe2)
    x23 = upsample(fiter, 4, 2)(xd3);     x24 = upsample2(fiter, 4)(xd4)
    x25 = upsample2(fiter, 8)(xd5);       x26 = upsample2(fiter, 16)(xe6)
    x27 = upsample2(fiter, 32)(xd7);      x28 = upsample2(fiter, 64)(xd8)
    x29 = upsample2(fiter, 128)(xe9)
    xd2 = tf.keras.layers.Concatenate()([x29, x28, x27, x26, x25, x24, x23, x22, x21])
    xd2 = conv(288)(xd2)

    x11 = downsample(fiter, 3, 1)(xe1);   x12 = upsample(fiter, 4, 2)(xd2)
    x13 = upsample2(fiter, 4)(xd3);       x14 = upsample2(fiter, 8)(xd4)
    x15 = upsample2(fiter, 16)(xd5);      x16 = upsample2(fiter, 32)(xe6)
    x17 = upsample2(fiter, 64)(xd7);      x18 = upsample2(fiter, 128)(xd8)
    x19 = upsample2(fiter, 256)(xe9)
    xd1 = tf.keras.layers.Concatenate()([x19, x18, x17, x16, x15, x14, x13, x12, x11])
    xd1 = conv(288)(xd1)

    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=1,
           padding='same', kernel_initializer=init, activation='tanh')
    x = last(xd1)
    return tf.keras.Model(inputs=[inp, energy_inp], outputs=x)

generator = Generator()
print(f"Generator 参数量: {generator.count_params():,}")

# ============================================================
# 7. Discriminator (PatchGAN, 三输入)
# ============================================================
def Discriminator():
    init = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_WIDTH, 3], name='input_image')
    energy = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_WIDTH, 3], name='energy_image')
    tar = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_WIDTH, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, energy, tar])
    down1 = downsample(64, 4, 2, False)(x)
    down2 = downsample(128, 4, 2)(down1)
    down3 = downsample(256, 4, 2)(down2)

    zp1 = tf.keras.layers.ZeroPadding2D()(down3)
    cv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=init,
          use_bias=False)(zp1)
    bn = tf.keras.layers.BatchNormalization()(cv)
    lr = tf.keras.layers.LeakyReLU()(bn)
    zp2 = tf.keras.layers.ZeroPadding2D()(lr)
    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=init)(zp2)

    return tf.keras.Model(inputs=[inp, energy, tar], outputs=last)

discriminator = Discriminator()
print(f"Discriminator 参数量: {discriminator.count_params():,}")

# ============================================================
# 8. 损失函数
# ============================================================
LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + LAMBDA * l1_loss
    return total_gen_loss, gan_loss, l1_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss, real_loss

generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.5)

# ============================================================
# 9. 检查点 (断点续训)
# ============================================================
gen_ckpt_path = os.path.join(SAVE_DIR, 'generator_checkpoint.weights.h5')
disc_ckpt_path = os.path.join(SAVE_DIR, 'discriminator_checkpoint.weights.h5')
loss_file_path = 'Loss.txt'

def get_start_epoch():
    if os.path.exists(loss_file_path):
        try:
            with open(loss_file_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    start = int(lines[-1].split()[0]) + 1
                    print(f"从 Loss.txt 恢复: epoch {start}")
                    return start
        except Exception as e:
            print(f"读取 Loss.txt 失败: {e}")
    return 0

try:
    generator.load_weights(gen_ckpt_path)
    discriminator.load_weights(disc_ckpt_path)
    print("已加载检查点权重。")
except Exception as e:
    print(f"未找到检查点 ({e})，从头开始训练。")

# ============================================================
# 10. 训练循环
# ============================================================
def generate_images(model, test_input, test_energy, tar, epoch, kj):
    prediction = model([test_input, test_energy], training=True)
    plt.figure(figsize=(20, 5))
    display_list = [test_input[0], test_energy[0], tar[0], prediction[0]]
    title = ['几何(输入)', '能量场(输入)', '裂纹(标准答案)', '裂纹(预测)']
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    out_path = os.path.join(OUT_DIR, f'epoch_{epoch:04d}_{kj}.png')
    plt.savefig(out_path, transparent=True)
    plt.close('all')

def train_step(input_image, energy_image, target, epoch, n):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator([input_image, energy_image], training=True)
        disc_real_output = discriminator([input_image, energy_image, target], training=True)
        disc_fake_output = discriminator([input_image, energy_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss, _ = generator_loss(
            disc_fake_output, gen_output, target)
        disc_loss, disc_real_loss = discriminator_loss(disc_real_output, disc_fake_output)

        # 每个 epoch 记录一次 (1600张/BATCH_SIZE=1 → n=0 到 1599, 记录在第1599步)
        total_steps = len(train_files) // BATCH_SIZE - 1
        if n == total_steps:
            with open(loss_file_path, 'a') as f:
                f.write(f"{epoch} {int(n)} {gen_l1_loss:.6f} {gen_l1_loss:.6f} "
                        f"{gen_gan_loss:.6f} {disc_loss:.6f} {disc_real_loss:.6f}\n")

    gen_grads = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

def fit(train_ds, epochs, test_ds, start_epoch):
    for epoch in range(start_epoch, epochs):
        # 每个 epoch 生成一张预览图
        for example_input, example_energy, example_target in test_ds.take(1):
            generate_images(generator, example_input, example_energy, example_target, epoch + 1, 1)

        print(f"Epoch {epoch}/{epochs} ", end='', flush=True)
        for n, (inp_img, eng_img, tar_img) in train_ds.enumerate():
            print('.', end='', flush=True)
            if (n + 1) % 100 == 0:
                print(f"[{n+1}]", end='', flush=True)
            train_step(inp_img, eng_img, tar_img, epoch, n)
        print(" done")

        # 每个 epoch 保存检查点
        generator.save_weights(gen_ckpt_path)
        discriminator.save_weights(disc_ckpt_path)
        print(f"  检查点已保存 → {SAVE_DIR}/")

# ============================================================
# 11. 启动训练
# ============================================================
print(f"\n{'='*60}")
print(f"BATCH_SIZE = {BATCH_SIZE} | 训练样本 = {len(train_files)} | 测试样本 = {len(test_files)}")
print(f"输出目录 = {OUT_DIR}/ | 检查点 = {SAVE_DIR}/")
print(f"{'='*60}\n")

EPOCHS = 1001
START_EPOCH = get_start_epoch()
print(f"开始训练: epoch {START_EPOCH} → {EPOCHS}")
fit(train_dataset, EPOCHS, test_dataset, START_EPOCH)
