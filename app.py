import gradio as gr
import torch
import torchaudio
import gc

from resemble_enhance.enhancer.inference import denoise, enhance

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def clear_gpu_cash():
    # del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _fn(path, solver, nfe, tau, chunk_seconds, chunks_overlap, denoising):
    if path is None:
        return None, None

    solver = solver.lower()
    nfe = int(nfe)
    lambd = 0.9 if denoising else 0.1

    dwav, sr = torchaudio.load(path)
    dwav = dwav.mean(dim=0)

    wav1, new_sr = denoise(dwav, sr, device)
    wav2, new_sr = enhance(
        dwav=dwav,
        sr=sr,
        device=device,
        nfe=nfe,
        chunk_seconds=chunk_seconds,
        chunks_overlap=chunks_overlap,
        solver=solver,
        lambd=lambd,
        tau=tau,
    )

    wav1 = wav1.cpu().numpy()
    wav2 = wav2.cpu().numpy()

    clear_gpu_cash()
    return (new_sr, wav1), (new_sr, wav2)


def main():
    inputs: list = [
        gr.Audio(type="filepath", label="输入音频"),
        gr.Dropdown(choices=["Midpoint", "RK4", "Euler"], value="Midpoint", label="CFM ODE 求解器（建议使用Midpoint）"),
        gr.Slider(
            minimum=1,
            maximum=128,
            value=64,
            step=1,
            label="CFM 函数评估次数（通常值越高质量越好，但速度可能越慢）",
        ),
        gr.Slider(
            minimum=0,
            maximum=1,
            value=0.5,
            step=0.01,
            label="CFM 先验温度（较高的值可以提高质量，但可能降低稳定性）",
        ),
        gr.Slider(minimum=1, maximum=40, value=10, step=1, label="单块秒数（单块时间越长，VRAM使用率越高）"),
        gr.Slider(minimum=0, maximum=5, value=1, step=0.5, label="块间交叠"),
        # chunk_seconds, chunks_overlap
        gr.Checkbox(value=False, label="增强前降噪（如果您的音频包含大量背景噪音，请勾选）"),
    ]

    outputs: list = [
        gr.Audio(label="输出降噪音频"),
        gr.Audio(label="输出增强音频"),
    ]

    interface = gr.Interface(
        fn=_fn,
        title="Resemble Enhance",
        description="由Resemble AI驱动的人工智能音频增强，为您的音频文件提供动力。",
        inputs=inputs,
        submit_btn="启动",
        clear_btn="重置",
        outputs=outputs,
        flagging_options=[("标记", "")],
    )

    interface.launch()


if __name__ == "__main__":
    main()
