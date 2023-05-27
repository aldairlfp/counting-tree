from keras import Input, Model
from keras import layers
from keras.optimizers import Adam

def unet_model(image_size):
    inputs = Input(shape=(image_size[0], image_size[1], image_size[2]))    # (width, height, bands) 380x380x8
    down1 = layers.Conv2D(64, (3, 3), activation='relu', padding ='same')(inputs)    # (378, 378, 64)
    down1 = layers.Conv2D(64, (3, 3), activation='relu', padding ='same')(down1)     # (376, 376, 64)
    
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(down1)                             # (188, 188, 64)
    down2 = layers.Conv2D(128, (3, 3), activation='relu', padding ='same')(pool1)    # (186, 186, 128)
    down2 = layers.Conv2D(128, (3, 3), activation='relu', padding ='same')(down2)    # (184, 184, 128)
    
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(down2)                             # (92, 92, 128)
    down3 = layers.Conv2D(256, (3, 3), activation='relu', padding ='same')(pool2)    # (90, 90, 256)
    down3 = layers.Conv2D(256, (3, 3), activation='relu', padding ='same')(down3)    # (88, 88, 256)
    
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(down3)                             # (44, 44, 256)
    down4 = layers.Conv2D(512, (3, 3), activation='relu', padding ='same')(pool3)    # (42, 42, 512)
    down4 = layers.Conv2D(512, (3, 3), activation='relu', padding ='same')(down4)    # (40, 40, 512)

    middle = layers.MaxPooling2D(pool_size=(2, 2))(down4)                            # (20, 20, 512)
    middle = layers.Conv2D(1024, (3, 3), activation='relu', padding ='same')(middle) # (18, 18, 1024)
    middle = layers.Conv2D(1024, (3, 3), activation='relu', padding ='same')(middle) # (16, 16, 1024)

    crop4 = layers.Cropping2D(4)(down4)                                              # (32, 32, 512)
    up4 = layers.UpSampling2D(size=(2, 2))(middle)                                   # (32, 32, 512)
    up4 = layers.concatenate([crop4, up4])                                           # (32, 32, 1024)
    up4 = layers.Conv2D(512, (3, 3), activation='relu', padding ='same')(up4)        # (30, 30, 512)
    up4 = layers.Conv2D(512, (3, 3), activation='relu', padding ='same')(up4)        # (28, 28, 512)

    crop3 = layers.Cropping2D(16)(down3)                                             # (56, 56, 256)
    up3 = layers.UpSampling2D(size=(2, 2))(up4)                                      # (56, 56, 256)
    up3 = layers.concatenate([crop3, up3])                                           # (56, 56, 512)
    up3 = layers.Conv2D(256, (3, 3), activation='relu', padding ='same')(up3)        # (54, 54, 256)
    up3 = layers.Conv2D(256, (3, 3), activation='relu', padding ='same')(up3)        # (52, 52, 256)

    crop2 = layers.Cropping2D(40)(down2)                                             # (104, 104, 128)
    up2 = layers.UpSampling2D(size=(2, 2))(up3)                                      # (104, 104, 128)
    up2 = layers.concatenate([crop2, up2])                                           # (104, 104, 256)
    up2 = layers.Conv2D(128, (3, 3), activation='relu')(up2)                         # (102, 102, 128)
    up2 = layers.Conv2D(128, (3, 3), activation='relu')(up2)                         # (100, 100, 128)

    crop1 = layers.Cropping2D(88)(down1)                                             # (200, 200, 64)
    up1 = layers.UpSampling2D(size=(2, 2))(up2)                                      # (200, 200, 64)
    up1 = layers.concatenate([crop1, up1])                                           # (200, 200, 128)
    up1 = layers.Conv2D(64, (3, 3), activation='relu')(up1)                          # (198, 198, 64)
    up1 = layers.Conv2D(64, (3, 3), activation='relu')(up1)                          # (196, 196, 64)

    model = layers.Conv2D(1, (1,1), activation='sigmoid')(up1)
    model = Model(inputs=inputs, outputs=model)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
    return model