import os
import io
import torch
import numpy as np
import base64
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.responses import JSONResponse, FileResponse # 这里新增了 FileResponse
from typing import List

# 导入您的网络和配置
from networks.vision_transformer import MMRSGUNet as ViT_seg
from config import get_config

# ==========================================
# 1. 初始化配置与模型 (保留您原本的逻辑)
# ==========================================
class MockArgs:
    dataset = 'Synapse'
    img_size = 224
    num_classes = 9
    cfg = 'configs/cswin_tiny_224_lite.yaml'
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
    batch_size = 1
    base_lr = 0.0001
    max_epochs = 250
    output_dir = './'
    list_dir = './lists/lists_Synapse'
    volume_path = '../data/Synapse'

args = MockArgs()
config = get_config(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).to(device)

model_path = '/home/chuanhaoyang/proiect_pratice_of_cnn/MMRSG-UNet-main/model/epoch_241.pth'
checkpoint = torch.load(model_path, map_location=device)

new_state_dict = {}
for k, v in checkpoint.items():
    new_key = k.replace('cswin_unet.', 'mmrsg_unet.').replace('MSCA', 'msdc')
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict, strict=False)
model.eval()

transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor()
])

# 类别名称与颜色映射
CLASS_NAMES = ["背景", "主动脉", "胆囊", "左肾", "右肾", "肝脏", "胰腺", "脾脏", "胃"]
COLORS = np.array([
    [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], 
    [255, 0, 255], [0, 255, 255], [255, 128, 0], [128, 0, 128],
], dtype=np.uint8)

# ==========================================
# 2. FastAPI 服务构建
# ==========================================
app = FastAPI(title="MMRSG-UNet Medical API")

# 允许跨域请求（方便前端调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# ==========================================
# 新增：让根目录直接返回我们的前端 HTML 网页
# ==========================================
@app.get("/")
async def serve_frontend():
    # 确保您的 index.html 和 app.py 放在同一个目录下
    return FileResponse("index.html")

@app.post("/predict")
async def predict_api(files: List[UploadFile] = File(...), alpha: float = Form(0.4)):
    try:
        images = []
        original_sizes = []
        
        # 1. 异步读取所有上传的图片
        for file in files:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            images.append(image)
            original_sizes.append(image.size)

        if not images:
            return JSONResponse(content={"status": "error", "message": "未接收到图片"}, status_code=400)

        # 2. 将多张图片堆叠成一个 Batch Tensor (B, C, H, W)
        tensor_list = [transform(img) for img in images]
        batch_tensor = torch.stack(tensor_list).to(device)

        # 3. 批量推理 (GPU 并行加速)
        with torch.no_grad():
            output = model(batch_tensor)
            # 输出形状应为 [B, num_classes, H, W]
            preds = output[0] if isinstance(output, list) else output
            # 批量获取类别索引，形状 [B, H, W]
            pred_masks = torch.argmax(preds, dim=1).cpu().numpy()

        # 4. 批量后处理与数据封装
        results = []
        for b in range(len(images)):
            pred_mask = pred_masks[b]
            original_size = original_sizes[b]
            filename = files[b].filename

            # 计算量化指标
            total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
            metrics = []
            for i in range(1, args.num_classes):
                pixel_count = np.sum(pred_mask == i)
                if pixel_count > 0:
                    percentage = (pixel_count / total_pixels) * 100
                    metrics.append({
                        "organ": CLASS_NAMES[i],
                        "pixel_count": int(pixel_count),
                        "percentage": f"{percentage:.2f}%"
                    })

            # 图像后处理与半透明叠加
            color_mask = COLORS[pred_mask]
            mask_image = Image.fromarray(color_mask).resize(original_size, Image.NEAREST)
            blend_image = Image.blend(images[b], mask_image, alpha=alpha)

            # 转换为 Base64
            buffered = io.BytesIO()
            blend_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            results.append({
                "filename": filename,
                "image_base64": f"data:image/png;base64,{img_str}",
                "metrics": metrics
            })

        return JSONResponse(content={
            "status": "success",
            "results": results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
    
async def predict_api(file: UploadFile = File(...), alpha: float = Form(0.4)):
    try:
        # 1. 读取并预处理图像
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        original_size = image.size

        # 2. 模型推理
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            pred = output[0] if isinstance(output, list) else output
            pred_mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

        # 3. 计算量化指标 (各个器官的像素面积占比)
        total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
        metrics = []
        for i in range(1, args.num_classes): # 跳过背景(0)
            pixel_count = np.sum(pred_mask == i)
            if pixel_count > 0:
                percentage = (pixel_count / total_pixels) * 100
                metrics.append({
                    "organ": CLASS_NAMES[i],
                    "pixel_count": int(pixel_count),
                    "percentage": f"{percentage:.2f}%"
                })

        # 4. 图像后处理与叠加
        color_mask = COLORS[pred_mask]
        mask_image = Image.fromarray(color_mask).resize(original_size, Image.NEAREST)
        blend_image = Image.blend(image, mask_image, alpha=alpha)

        # 5. 转换为 Base64 以供前端渲染
        buffered = io.BytesIO()
        blend_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return JSONResponse(content={
            "status": "success",
            "image_base64": f"data:image/png;base64,{img_str}",
            "metrics": metrics
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
