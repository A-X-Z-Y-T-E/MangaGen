import os
import argparse
import json
from typing import List, Dict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

class PanelPrompts(BaseModel):
    """Model for panel prompts output"""
    panels: List[str] = Field(description="List of panel descriptions")

class MangaPromptGenerator:
    """
    Generate panel-specific prompts for manga using Groq LLM API via LangChain
    """
    
    def __init__(self, api_key=None):
        # Use API key from .env file or passed parameter
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass api_key parameter.")
        
    def generate_panel_prompts(self, 
                             story_prompt: str, 
                             num_panels: int = 4, 
                             style: str = "manga", 
                             model: str = "llama3-70b-8192") -> List[str]:
        """
        Generate prompts for each panel of a manga based on a main story prompt
        
        Args:
            story_prompt: Main story description/prompt
            num_panels: Number of panels to generate prompts for
            style: The manga style to aim for
            model: Groq model to use
            
        Returns:
            List of panel-specific prompts
        """
        
        # Initialize the LangChain ChatGroq model
        chat = ChatGroq(
            temperature=0.7,
            groq_api_key=self.api_key,
            model_name=model
        )
        
        system_prompt = f"""You are a professional manga artist and storyteller. 
Given a main story prompt, break it down into {num_panels} sequential manga panels.
For each panel, create a concise visual description that will be used for image generation.
Focus on {style} style visuals, character expressions, composition, action, and mood.

IMPORTANT: Keep each panel description between 30-70 words maximum. 
The image generation model has a strict token limit of 77 tokens per prompt.
Be brief but specific, focusing on the most important visual elements.

The panels should flow together to tell a cohesive visual story.
Your response should be in JSON format with a 'panels' field containing an array of panel descriptions."""
        
        user_prompt = f"""Main story prompt: {story_prompt}
        
Please create {num_panels} sequential panel descriptions that tell this story in {style} style.
IMPORTANT: Keep each panel description BRIEF (30-70 words maximum) to fit within token limits.
Focus on core visual elements: composition, character positions, expressions, key background elements.
Make each panel visually distinct while maintaining narrative flow."""
        
        try:
            # Create messages for the chat model
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # Get response from the model
            try:
                # Use invoke method (newer LangChain version)
                response = chat.invoke(messages)
                
                # Debug the response type
                print(f"Raw response type: {type(response)}")
                print(f"Response attributes: {dir(response)[:10]}...")
                
                # Extract content from different possible response types
                content = None
                
                # New LangChain Content message type
                if hasattr(response, 'content'):
                    content = response.content
                    print("Using response.content")
                
                # LangChain v0.1+ AIMessage structures
                elif hasattr(response, 'message') and hasattr(response.message, 'content'):
                    content = response.message.content
                    print("Using response.message.content")
                    
                # LangChain v0.0.x chat message
                elif hasattr(response, 'text'):
                    content = response.text
                    print("Using response.text")
                    
                # If it's a dict with content key
                elif isinstance(response, dict) and 'content' in response:
                    content = response['content']
                    print("Using response['content']")
                    
                # If it's an older LangChain message format
                elif hasattr(response, '__getitem__') and len(response) > 0:
                    try:
                        first_msg = response[0]
                        if hasattr(first_msg, 'content'):
                            content = first_msg.content
                            print("Using response[0].content")
                        elif hasattr(first_msg, 'text'):
                            content = first_msg.text
                            print("Using response[0].text")
                    except (IndexError, TypeError):
                        pass
                    
                # Last resort: try to convert response to string
                if content is None:
                    print("Fallback: converting response to string")
                    content = str(response)
                    # If it looks like it might be a Message object converted to string
                    if "{" in content and "}" in content:
                        try:
                            # Try to extract JSON from string representation
                            start_idx = content.find("{")
                            end_idx = content.rfind("}") + 1
                            json_str = content[start_idx:end_idx]
                            data = json.loads(json_str)
                            if "content" in data:
                                content = data["content"]
                                print("Extracted content from JSON representation")
                        except json.JSONDecodeError:
                            pass
                
            except Exception as invoke_error:
                print(f"Warning: Error invoking chat model: {invoke_error}")
                print("Trying alternative approach with __call__...")
                # Alternative approach using __call__ for older LangChain versions
                try:
                    response = chat(messages)
                    if hasattr(response, 'content'):
                        content = response.content
                        print("Using __call__ response.content")
                    else:
                        content = str(response)
                        print("Using __call__ str(response)")
                except Exception as call_error:
                    raise Exception(f"Failed to get response from chat model: {call_error}")
            
            # Make sure we have a string content
            if not isinstance(content, str):
                print(f"Warning: content is not a string, converting from {type(content)}")
                content = str(content)
            
            print(f"Content excerpt: {content[:100]}...")
            
            # Multiple parsing strategies
            try:
                # Try to parse as JSON directly
                if "```json" in content:
                    # Extract JSON if it's in a code block
                    start = content.find("```json") + 7
                    end = content.find("```", start)
                    if start > 6 and end > start:
                        content = content[start:end].strip()
                        print("Extracted JSON from code block")
                elif "```" in content:
                    # Try generic code block
                    start = content.find("```") + 3
                    # Skip language identifier if present
                    if content[start:start+10].strip() and "\n" in content[start:start+20]:
                        start = content.find("\n", start) + 1
                    end = content.find("```", start)
                    if start > 3 and end > start:
                        content = content[start:end].strip()
                        print("Extracted content from generic code block")
                
                # Try to parse the JSON
                try:
                    panel_data = json.loads(content)
                    print(f"Successfully parsed JSON data type: {type(panel_data)}")
                    
                    # Handle different JSON formats
                    if isinstance(panel_data, list):
                        prompts = panel_data
                        print("Using list of prompts directly")
                    elif isinstance(panel_data, dict):
                        if "panels" in panel_data:
                            prompts = panel_data["panels"]
                            print("Using panels field from dict")
                        else:
                            # Assume it's a dict with numbered keys or panel keys
                            prompts = list(panel_data.values())
                            print("Using dict values as prompts")
                    else:
                        raise ValueError(f"Unexpected JSON data type: {type(panel_data)}")
                
                    # Make sure all prompts are within length limits
                    processed_prompts = []
                    for i, prompt in enumerate(prompts):
                        if not isinstance(prompt, str):
                            print(f"Converting prompt {i} from {type(prompt)} to string")
                            prompt = str(prompt)
                        # Limit to around 200 chars (conservative estimate for ~50 tokens)
                        if len(prompt) > 200:
                            prompt = prompt[:200] + "..."
                        processed_prompts.append(prompt)
                    
                    print(f"Final number of prompts: {len(processed_prompts)}")
                    return processed_prompts
                
                except json.JSONDecodeError as json_error:
                    print(f"JSON parsing failed: {json_error}. Trying text-based parsing.")
                    # If not valid JSON, try text-based parsing
                    if "Panel 1:" in content:
                        # Parse text format with "Panel X:" prefixes
                        panel_prompts = []
                        for i in range(1, num_panels + 1):
                            marker = f"Panel {i}:"
                            next_marker = f"Panel {i+1}:" if i < num_panels else None
                            
                            start = content.find(marker)
                            if start != -1:
                                start += len(marker)
                                end = content.find(next_marker) if next_marker else len(content)
                                panel_text = content[start:end].strip()
                                
                                # Enforce length limit
                                if len(panel_text) > 200:
                                    panel_text = panel_text[:200] + "..."
                                    
                                panel_prompts.append(panel_text)
                        
                        if panel_prompts:
                            print(f"Extracted {len(panel_prompts)} panels using 'Panel X:' format")
                            return panel_prompts
                    
                    # Last resort: Split content into roughly equal parts
                    print("Using last resort text splitting approach")
                    lines = content.split('\n')
                    non_empty_lines = [line for line in lines if line.strip()]
                    if len(non_empty_lines) >= num_panels:
                        # If we have at least as many lines as panels, use one line per panel
                        result = non_empty_lines[:num_panels]
                        print(f"Using {len(result)} non-empty lines as panel prompts")
                        return result
                    else:
                        # Otherwise split the content into equal chunks
                        chunk_size = len(content) // num_panels
                        panel_prompts = []
                        for i in range(num_panels):
                            start = i * chunk_size
                            end = start + chunk_size if i < num_panels - 1 else len(content)
                            chunk = content[start:end].strip()
                            
                            # Enforce length limit
                            if len(chunk) > 200:
                                chunk = chunk[:200] + "..."
                                
                            panel_prompts.append(chunk)
                        print(f"Split content into {len(panel_prompts)} equal chunks")
                        return panel_prompts
            
            except Exception as parsing_error:
                # If all parsing attempts fail
                raise ValueError(f"Failed to parse panel prompts from response: {parsing_error}")
                
        except Exception as e:
            raise Exception(f"Error generating panel prompts: {str(e)}")
    
    def save_prompts_to_file(self, prompts: List[str], output_file: str = "panel_prompts.json"):
        """Save generated panel prompts to a JSON file"""
        with open(output_file, 'w') as f:
            json.dump(prompts, f, indent=2)
        
        return output_file


def parse_args():
    parser = argparse.ArgumentParser(description="Generate manga panel prompts using Groq LLM API with LangChain")
    parser.add_argument("--prompt", type=str, required=True, 
                        help="Main story prompt/description for the manga")
    parser.add_argument("--panels", type=int, default=4,
                        help="Number of panels to generate prompts for")
    parser.add_argument("--style", type=str, default="manga",
                        help="Style of manga to generate (e.g., 'shonen', 'seinen', 'shoujo')")
    parser.add_argument("--output", type=str, default="panel_prompts.json",
                        help="Output file path for panel prompts")
    parser.add_argument("--model", type=str, default="llama3-70b-8192",
                        help="Groq model to use (llama3-70b-8192, mixtral-8x7b, etc.)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize generator
    try:
        generator = MangaPromptGenerator()
        
        print(f"Generating prompts for {args.panels} manga panels...")
        panel_prompts = generator.generate_panel_prompts(
            story_prompt=args.prompt,
            num_panels=args.panels,
            style=args.style,
            model=args.model
        )
        
        # Save prompts to file
        output_file = generator.save_prompts_to_file(panel_prompts, args.output)
        print(f"Generated {len(panel_prompts)} panel prompts and saved to {output_file}")
        
        # Print the prompts
        for i, prompt in enumerate(panel_prompts, 1):
            print(f"\nPanel {i}:")
            print(prompt[:100] + "..." if len(prompt) > 100 else prompt)
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main() 