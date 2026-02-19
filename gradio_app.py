"""
Gradio Web Interface for Fine-Tuned LLM
Provides an intuitive UI for interacting with the fine-tuned model.
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os


class ModelInterface:
    """Wrapper class for model interaction."""
    
    def __init__(self, base_model_name: str, adapter_path: str = None, use_4bit: bool = True):
        """
        Initialize model interface.
        
        Args:
            base_model_name: Base model name
            adapter_path: Path to LoRA adapters (if fine-tuned)
            use_4bit: Whether to use 4-bit quantization
        """
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.use_4bit = use_4bit
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load model and tokenizer."""
        print(f"Loading model: {self.base_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load base model
        if self.use_4bit:
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        
        # Load LoRA adapters if provided
        if self.adapter_path and os.path.exists(self.adapter_path):
            print(f"Loading LoRA adapters from: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            self.model.eval()
    
    def generate_response(self, instruction: str, max_tokens: int = 256, 
                         temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate response from model.
        
        Args:
            instruction: User instruction/question
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated response
        """
        if self.model is None or self.tokenizer is None:
            return "Model not loaded. Please check model path."
        
        # Format prompt (Alpaca format)
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response part
        response = response.split("### Response:")[-1].strip()
        
        return response


def create_interface(base_model_name: str, adapter_path: str = None, 
                    use_4bit: bool = True, title: str = "OrthoAssist - Orthopedic Medicine Assistant"):
    """
    Create Gradio interface.
    
    Args:
        base_model_name: Base model name
        adapter_path: Path to LoRA adapters
        use_4bit: Whether to use 4-bit quantization
        title: Interface title
        
    Returns:
        Gradio Interface object
    """
    # Initialize model
    model_interface = ModelInterface(base_model_name, adapter_path, use_4bit)
    
    # Example queries
    examples = [
        "What are the different types of bone fractures?",
        "How long does it take for a broken arm to heal?",
        "What is the treatment for a compound fracture?",
        "What are the symptoms of a stress fracture?",
        "How do you prevent bone fractures?",
        "What is the difference between a fracture and a break?",
        "What should I do if I suspect I have a broken bone?",
        "What is physical therapy for fracture recovery?"
    ]
    
    # Define interface function
    def chat(instruction, max_tokens, temperature, top_p, history):
        """Chat function for Gradio."""
        response = model_interface.generate_response(
            instruction, max_tokens, temperature, top_p
        )
        
        # Add medical disclaimer
        disclaimer = "\n\n⚠️ **Medical Disclaimer**: This information is for educational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult a healthcare professional for medical concerns."
        
        return response + disclaimer
    
    # Create interface
    with gr.Blocks(title=title, theme=gr.themes.Soft()) as interface:
        gr.Markdown(f"# {title}\n\n**OrthoAssist** - Your AI assistant for orthopedic medicine, bone fractures, and musculoskeletal health.\n\nAsk questions and get accurate, educational information about orthopedic conditions, treatments, and recovery.")
        
        with gr.Row():
            with gr.Column(scale=2):
                instruction_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Enter your question about bone fractures or orthopedic medicine...",
                    lines=3
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    max_tokens = gr.Slider(
                        minimum=50,
                        maximum=512,
                        value=256,
                        step=50,
                        label="Max Tokens"
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.5,
                        value=0.7,
                        step=0.1,
                        label="Temperature"
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Top-p (Nucleus Sampling)"
                    )
                
                submit_btn = gr.Button("Submit", variant="primary", size="lg")
            
            with gr.Column(scale=3):
                output = gr.Textbox(
                    label="Response",
                    lines=10,
                    interactive=False
                )
        
        gr.Markdown("### Example Questions")
        gr.Examples(
            examples=examples,
            inputs=instruction_input
        )
        
        gr.Markdown("""
        ### Instructions
        1. Enter your question in the text box
        2. Adjust advanced settings if needed (optional)
        3. Click Submit to get a response
        4. Try the example questions below for inspiration
        
        ### About
        This assistant is fine-tuned on medical literature about bone fractures and orthopedic medicine.
        It provides educational information but should not replace professional medical advice.
        """)
        
        # Connect submit button
        submit_btn.click(
            fn=chat,
            inputs=[instruction_input, max_tokens, temperature, top_p, None],
            outputs=output
        )
        
        # Also allow Enter key
        instruction_input.submit(
            fn=chat,
            inputs=[instruction_input, max_tokens, temperature, top_p, None],
            outputs=output
        )
    
    return interface


def launch_app(base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
              adapter_path: str = None,
              use_4bit: bool = True,
              server_name: str = "0.0.0.0",
              server_port: int = 7860,
              share: bool = False):
    """
    Launch Gradio app.
    
    Args:
        base_model_name: Base model name
        adapter_path: Path to LoRA adapters
        use_4bit: Whether to use 4-bit quantization
        server_name: Server name
        server_port: Server port
        share: Whether to create public link
    """
    interface = create_interface(base_model_name, adapter_path, use_4bit)
    interface.launch(
        server_name=server_name,
        server_port=server_port,
        share=share
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Gradio interface for fine-tuned LLM")
    parser.add_argument("--base_model", type=str, 
                       default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                       help="Base model name")
    parser.add_argument("--adapter_path", type=str, default=None,
                       help="Path to LoRA adapters")
    parser.add_argument("--use_4bit", action="store_true",
                       help="Use 4-bit quantization")
    parser.add_argument("--port", type=int, default=7860,
                       help="Server port")
    parser.add_argument("--share", action="store_true",
                       help="Create public link")
    
    args = parser.parse_args()
    
    launch_app(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        use_4bit=args.use_4bit,
        server_port=args.port,
        share=args.share
    )
