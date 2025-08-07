import gradio as gr
import yaml
from datetime import datetime
import json
from typing import List, Dict
from orchestrator import RecommenderOrchestrator

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Instantiate orchestrator
orchestrator = RecommenderOrchestrator("config.yaml")

def chat_turn_with_progress(message, chat_history, progress=gr.Progress()):
    """Chat function with real-time progress updates"""
    rec_context = {"recommendations_shown": bool(chat_history and "Recommendations:" in chat_history[-1][1])}
    
    def update_progress(value, desc):
        progress(value, desc=desc)
    
    resp = orchestrator.handle_chat_with_progress_steps(message, rec_context, update_progress)
    
    # Build response
    bot_text = resp["response"]
    if "recommendations" in resp:
        items = resp["recommendations"]
        formatted = "\n".join(f"- {item['food_name']} from {item.get('metadata', {}).get('restaurant','')} |{item['metadata'].get('r_rating')} food rating: ⭐ {item['metadata'].get('f_rating')} price : {item['metadata'].get('f_price')} \n " for item in items)
        bot_text = "\n\n🍽️ **Recommendations:**\n" + formatted
    
    chat_history.append((message, bot_text))
    return "", chat_history

def get_history_data():
    """Get formatted history data"""
    history = orchestrator.format_history_for_display()
    if not history:
        return "No search history available yet."
    
    history_display = []
    for item in history:
        history_display.append(f"**{item['readable_time']}** - {item['preview']}")
    
    return "\n\n".join(history_display)


def show_history_details(selection):
    """Show raw JSON details for selected history item"""
    if not selection or "No search history" in selection:
        return "No history selected", ""
    
    try:
        # Extract timestamp to find the right history item
        timestamp_line = selection.split('\n')[0]
        history = orchestrator.get_search_history()
        
        # Find matching history item
        for i, item in enumerate(history):
            if orchestrator._format_timestamp(item["timestamp"]) in timestamp_line:
                # Return complete JSON structure
                formatted_json = json.dumps(item, indent=2, ensure_ascii=False)
                return f"``````", ""
                
    except Exception as e:
        return f"Error loading history details: {str(e)}", ""
    
    return "History item not found", ""


def refresh_history():
    """Refresh history display"""
    return get_history_data()

# Create the Gradio interface
with gr.Blocks(title="🍽️ AI Food Recommender") as demo:
    
    gr.Markdown("# 🍽️ AI Food Recommender Chatbot")
    
    # Navigation tabs
    with gr.Tabs() as nav_tabs:
        
        # Chat Tab
        with gr.Tab("💬 Chat", elem_id="chat_tab"):
            
            chatbot = gr.Chatbot(height=500)
            
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Type your message here… (Press Enter to send)",
                    scale=4,
                    show_label=False
                )
                send_button = gr.Button("Send", scale=1, variant="primary")
            
            # Chat functionality
            user_input.submit(
                chat_turn_with_progress,
                inputs=[user_input, chatbot],
                outputs=[user_input, chatbot],
            )
            
            send_button.click(
                chat_turn_with_progress,
                inputs=[user_input, chatbot],
                outputs=[user_input, chatbot],
            )
        
        # History Tab
        # History Tab
        with gr.Tab("📚 Search History", elem_id="history_tab"):
            gr.Markdown("### View your previous search results (Raw JSON)")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Search Timeline")
                    refresh_btn = gr.Button("🔄 Refresh History", variant="secondary")
                    
                    history_display = gr.Markdown(
                        value=get_history_data(),
                        label="Search History"
                    )
                    
                    history_selector = gr.Dropdown(
                        choices=[],
                        label="Select a search to view JSON",
                        interactive=True
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("#### Raw JSON Data")
                    json_display = gr.Code(
                        value="Select a search from the timeline to view raw JSON data",
                        language="json",
                        label="Complete Search Data",
                        lines=20
                    )
            
            # History functionality with JSON display
            def update_history_and_show_json():
                history = orchestrator.format_history_for_display()
                choices = [f"{item['readable_time']} - {item['preview']}" for item in history]
                return gr.Dropdown(choices=choices), get_history_data()
            
            def show_json_for_selection(selection):
                if not selection or "No search history" in selection:
                    return "// No history selected"
                
                try:
                    timestamp_line = selection.split(' - ')[0]
                    history = orchestrator.get_search_history()
                    
                    for item in history:
                        if orchestrator._format_timestamp(item["timestamp"]) in timestamp_line:
                            return json.dumps(item, indent=2, ensure_ascii=False)
                            
                except Exception as e:
                    return f"// Error: {str(e)}"
                
                return "// History item not found"
            
            refresh_btn.click(
                fn=update_history_and_show_json,
                outputs=[history_selector, history_display]
            )
            
            history_selector.change(
                fn=show_json_for_selection,
                inputs=[history_selector],
                outputs=[json_display]
            )

    def update_history_selector():
        """Update dropdown choices with current history"""
        history = orchestrator.format_history_for_display()
        if not history:
            return gr.Dropdown(choices=[], value=None)
    
        choices = [f"{item['readable_time']} - {item['preview']}" for item in history]
        return gr.Dropdown(choices=choices, value=None)


    # Auto-refresh history when page loads
    demo.load(
        fn=lambda: update_history_selector(),
        outputs=[history_selector]
    )

if __name__ == "__main__":
    demo.launch()
