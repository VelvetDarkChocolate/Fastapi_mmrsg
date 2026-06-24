import os
import torch
import numpy as np
import gradio as gr
from PIL import Image
from torchvision import transforms

# 导入 MMRSG-UNet 专属网络和配置 (基于你项目里的 test.py 写法)
from networks.vision_transformer import MMRSGUNet as ViT_seg
from config import get_config

# ==========================================
# 1. 模拟命令行参数 (为了无需命令行传参直接运行网页)
# ==========================================
class MockArgs:
    dataset = 'Synapse'
    img_size = 224
    num_classes = 9
    cfg = 'configs/cswin_tiny_224_lite.yaml'  # 根据你实际的配置文件修改
    opts = None
    zip = False
    cache_mode = 'part'
    resume = None
    accumulation_steps = None
    use_checkpoint = False
    amp_opt_level = 'O1'
    tag = None
    eval = False
    throughput = False
    
    # --- 新增的属性，防止 config.py 报错 ---
    batch_size = 1       # 推理时 batch_size 设为 1 即可
    base_lr = 0.0001
    max_epochs = 250
    output_dir = './'    # 防止找不到输出目录报错
    list_dir = './lists/lists_Synapse'
    volume_path = '../data/Synapse'

args = MockArgs()
config = get_config(args)

# ==========================================
# 2. 初始化并加载 MMRSG-UNet 模型
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).to(device)

# 填入你提供的权重绝对路径
model_path = '/home/chuanhaoyang/proiect_pratice_of_cnn/MMRSG-UNet-main/model/epoch_241.pth'
checkpoint = torch.load(model_path, map_location=device)

# --- 核心修复：处理版本迭代导致的网络层改名问题 ---
new_state_dict = {}
for k, v in checkpoint.items():
    # 1. 修复最外层网络改名: cswin_unet -> mmrsg_unet
    new_key = k.replace('cswin_unet.', 'mmrsg_unet.')
    
    # 2. 修复内部模块改名: MSCA -> msdc
    new_key = new_key.replace('MSCA', 'msdc')
    
    new_state_dict[new_key] = v

# 使用修复后的字典加载权重
# strict=False 允许忽略掉原来模型里有，但现在代码里已经被删掉的冗余层 (如 conv33conv33conv11)
model.load_state_dict(new_state_dict, strict=False)
model.eval() # 设置为推理模式

# ==========================================
# 3. 图像预处理与色彩映射表
# ==========================================
transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor()
])

# 定义 Synapse 数据集 9 个类别的渲染颜色 (RGB格式)
# 0是背景(黑), 1-8是各种器官颜色
COLORS = np.array([
    [0, 0, 0],         # Background: 黑
    [255, 0, 0],       # Class 1: 红
    [0, 255, 0],       # Class 2: 绿
    [0, 0, 255],       # Class 3: 蓝
    [255, 255, 0],     # Class 4: 黄
    [255, 0, 255],     # Class 5: 洋红
    [0, 255, 255],     # Class 6: 青色
    [255, 128, 0],     # Class 7: 橙色
    [128, 0, 128],     # Class 8: 紫色
], dtype=np.uint8)

# ==========================================
# 4. 核心推理与后处理函数
# ==========================================
def predict(image):
    try:
        if image is None:
            return None
            
        # 1. 容错处理与通道转换
        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        if hasattr(image, 'mode') and image.mode != 'RGB':
            image = image.convert('RGB')
            
        original_size = image.size # 记录原图大小以便复原

        # 2. 推理计算
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            # mmrsg_unet 的 forward 函数返回了一个列表 [p1, p2, p3, p4]
            # p1 是最终的上采样输出结果
            pred = output[0] if isinstance(output, list) else output
            
            # 在 Channel 维度(dim=1)上取 argmax 获取每个像素所属的类别索引
            pred_mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

        # 3. 后处理：将数字索引 (0~8) 映射为实际的 RGB 颜色图片
        color_mask = COLORS[pred_mask]
        mask_image = Image.fromarray(color_mask)
        
        # 将 Mask 放大回用户上传图片的长宽比例
        mask_image = mask_image.resize(original_size, Image.NEAREST)

        # 将原图与 Mask 半透明叠加 (alpha=0.4表示透明度)，方便直观查看器官位置
        blend_image = Image.blend(image, mask_image, alpha=0.4)

        return blend_image
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        # 万一报错，返回纯黑图片或者抛出错误方便调试
        return image

# ==========================================
# 5. 启动 Gradio 网页
# ==========================================
interface = gr.Interface(
    fn=predict,                                       
    inputs=gr.Image(type="pil", label="上传医学扫描图片 (如 MRI/CT)"),  
    outputs=gr.Image(type="pil", label="AI 器官分割结果"), 
    title="MMRSG-UNet 医疗图像分割平台",
    description="上传您的扫描图像，模型将自动分割出病灶及不同的器官组织。模型权重使用: epoch_241.pth"
)

if __name__ == "__main__":
    # share=True 可以让你生成一个外网可访问的公共链接
    interface.launch(share=True, server_name="0.0.0.0", server_port=7860)