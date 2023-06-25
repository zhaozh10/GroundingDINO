import albumentations as A

# 定义第一种分辨率的 Resize 变换
resize_512 = A.Resize(512, 512)

# 定义第二种分辨率的 Resize 变换
resize_1024 = A.Resize(1024, 1024)

# 定义一系列共享的增强变换
shared_transforms = A.Compose([
    # 在这里添加你的其他变换
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    # ...
])

# 定义 transform，根据需要选择不同的 Resize 变换
transform = A.Compose([
    A.OneOf([
        resize_512,
        resize_1024,
    ]),
    shared_transforms,
])
