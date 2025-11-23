# from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
import os

def save_graph_visualization(graph, output_dir="overlay"):
    """Save graph visualization as PNG using IPython display"""
    try:
        # Create visualizations directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print("Output directory checked or created.")

        # Check if graph.get_graph() itself may be causing a delay
        graph_obj = graph.get_graph()
        print("Graph object retrieved.")

        print("Attempting to draw PNG...")
        # Generate PNG using Mermaid
        png_data = graph.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
            curve_style=CurveStyle.LINEAR,
            node_colors=NodeStyles(
                first="#ffdfba",  # Start node
                last="#baffc9",   # End node
                default="#f2f0ff"  # Other nodes
            ),
            wrap_label_n_words=4,
            background_color="white",
            padding=10
        )

        print("PNG drawn successfully.")
        
        # Save PNG file
        output_path = os.path.join(output_dir, "agent_graph.png")
        with open(output_path, "wb") as f:
            f.write(png_data)
            
        # # Display the image if in IPython environment
        # try:
        #     display(Image(png_data))
        # except:
        #     logger.info("Not in IPython environment - skipping display")
            
    except Exception as e:
        print(f"Error saving graph visualization: {str(e)}")