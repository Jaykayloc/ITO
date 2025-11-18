import gradio as gr
import torch
from ITO import ITOPipeline, make_grid

# Initialize the pipeline
print("Loading ITO Pipeline...")
ito = ITOPipeline()
print("Pipeline loaded!")

def generate_images(
    prompt,
    negative_prompt,
    use_ito,
    budget,
    lambda_max,
    alpha,
    guidance_scale,
    num_steps,
    width,
    height,
    seed,
):
    try:
        if use_ito:
            image, total_kl, lambdas = ito.generate_ito(
                prompt=prompt,
                negative_prompt=negative_prompt,
                budget=budget,
                lambda_max=lambda_max,
                num_steps=num_steps,
                height=height,
                width=width,
                alpha=alpha,
                seed=seed,
                verbose=False,
            )
            info = f"ITO Generation\nTotal KL: {total_kl:.2f}\nAvg Lambda: {sum(lambdas)/len(lambdas):.2f}"
            return image, info
        else:
            image = ito.generate_fixed(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                height=height,
                width=width,
                seed=seed,
            )
            info = f"Standard SDXL\nCFG Scale: {guidance_scale}"
            return image, info
    except Exception as e:
        return None, f"Error: {str(e)}"

def generate_comparison(
    prompt,
    negative_prompt,
    budget,
    lambda_max,
    alpha,
    guidance_scale,
    num_steps,
    width,
    height,
    seed,
):
    try:
        image_ito, total_kl, lambdas = ito.generate_ito(
            prompt=prompt,
            negative_prompt=negative_prompt,
            budget=budget,
            lambda_max=lambda_max,
            num_steps=num_steps,
            height=height,
            width=width,
            alpha=alpha,
            seed=seed,
            verbose=False,
        )
        
        image_cfg = ito.generate_fixed(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            height=height,
            width=width,
            seed=seed,
        )
        
        grid = make_grid(
            images=[image_ito, image_cfg],
            labels=[f"ITO (KL={total_kl:.1f})", f"CFG={guidance_scale}"],
        )
        
        info = f"Comparison Generated\nITO Total KL: {total_kl:.2f}\nITO Avg Lambda: {sum(lambdas)/len(lambdas):.2f}"
        return grid, info
    except Exception as e:
        return None, f"Error: {str(e)}"

# Gradio interface
with gr.Blocks(title="ITO Image Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎨 ITO: Information-Theoretic Optimization for Stable Diffusion XL
    
    Generate images using adaptive per-step guidance (ITO) or standard fixed CFG scaling.
    
    **ITO** adjusts guidance strength dynamically based on a KL budget, typically producing better results than fixed CFG.
    """)
    
    with gr.Tab("Single Generation"):
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="a cozy cabin in a snowy forest, warm lights in the windows",
                    lines=3,
                )
                negative_prompt_input = gr.Textbox(
                    label="Negative Prompt",
                    value="low quality, blurry, distorted, text, watermark",
                    lines=2,
                )
                
                use_ito = gr.Checkbox(label="Use ITO (vs. Standard CFG)", value=True)
                
                with gr.Row():
                    seed_input = gr.Number(label="Seed", value=42, precision=0)
                    num_steps_input = gr.Slider(label="Steps", minimum=20, maximum=50, value=30, step=1)
                
                with gr.Row():
                    width_input = gr.Slider(label="Width", minimum=512, maximum=1024, value=1024, step=64)
                    height_input = gr.Slider(label="Height", minimum=512, maximum=1024, value=1024, step=64)
                
                with gr.Accordion("ITO Parameters", open=False):
                    budget_input = gr.Slider(label="Budget", minimum=10, maximum=1000, value=40, step=5)
                    lambda_max_input = gr.Slider(label="Lambda Max", minimum=5, maximum=15, value=7.5, step=0.5)
                    alpha_input = gr.Slider(label="Alpha (rescale)", minimum=0, maximum=1, value=0.3, step=0.05)
                
                with gr.Accordion("Standard CFG Parameters", open=False):
                    guidance_scale_input = gr.Slider(label="Guidance Scale", minimum=1, maximum=15, value=7.5, step=0.5)
                
                generate_btn = gr.Button("Generate Image", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(label="Generated Image", type="pil")
                output_info = gr.Textbox(label="Generation Info", lines=3)
        
        generate_btn.click(
            fn=generate_images,
            inputs=[
                prompt_input, negative_prompt_input, use_ito,
                budget_input, lambda_max_input, alpha_input,
                guidance_scale_input, num_steps_input,
                width_input, height_input, seed_input
            ],
            outputs=[output_image, output_info],
        )
    
    with gr.Tab("Compare ITO vs CFG"):
        with gr.Row():
            with gr.Column():
                comp_prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="a futuristic cityscape at dusk, cinematic lighting",
                    lines=3,
                )
                comp_negative = gr.Textbox(
                    label="Negative Prompt",
                    value="low quality, blurry, distorted, text, watermark",
                    lines=2,
                )
                
                with gr.Row():
                    comp_seed = gr.Number(label="Seed", value=42, precision=0)
                    comp_steps = gr.Slider(label="Steps", minimum=20, maximum=50, value=30, step=1)
                
                with gr.Row():
                    comp_width = gr.Slider(label="Width", minimum=512, maximum=1024, value=1024, step=64)
                    comp_height = gr.Slider(label="Height", minimum=512, maximum=1024, value=1024, step=64)
                
                with gr.Row():
                    comp_budget = gr.Slider(label="ITO Budget", minimum=10, maximum=1000, value=40, step=5)
                    comp_guidance = gr.Slider(label="CFG Scale", minimum=1, maximum=15, value=7.5, step=0.5)
                
                with gr.Row():
                    comp_lambda = gr.Slider(label="Lambda Max", minimum=5, maximum=15, value=7.5, step=0.5)
                    comp_alpha = gr.Slider(label="Alpha", minimum=0, maximum=1, value=0.3, step=0.05)
                
                compare_btn = gr.Button("Generate Comparison", variant="primary")
            
            with gr.Column():
                comp_output = gr.Image(label="Comparison (ITO | CFG)", type="pil")
                comp_info = gr.Textbox(label="Comparison Info", lines=3)
        
        compare_btn.click(
            fn=generate_comparison,
            inputs=[
                comp_prompt, comp_negative,
                comp_budget, comp_lambda, comp_alpha,
                comp_guidance, comp_steps,
                comp_width, comp_height, comp_seed
            ],
            outputs=[comp_output, comp_info],
        )
    
    gr.Markdown("""
    ### About ITO
    
    Information-Theoretic Optimization treats each diffusion step as spending part of a KL budget:
    - **Early steps** (high noise) get higher guidance strength
    - **Later steps** (low noise) get lower guidance strength
    - Control overall guidance via a single **budget** parameter instead of fixed CFG scale
    
    **Tips:**
    - Start with budget=30-50, lambda_max=7.5, alpha=0.3
    - Increase budget for stronger prompt adherence
    - Increase alpha for more stability
    
    ---
    
    Based on: [ITO GitHub Repository](https://github.com/leochlon/ITO)
    """)

if __name__ == "__main__":
    demo.launch()