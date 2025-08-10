from jinja2 import Environment, FileSystemLoader, select_autoescape
import os

# Set up Jinja2 environment
env = Environment(
    loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), 'templates')),
    autoescape=select_autoescape(['html', 'xml'])
)

template = env.get_template('index.html')

# Dummy style data to simulate style.json
dummy_style = {
    "location": {"query": "Test", "distance": None},
    "layers": {
        "streets": {"enabled": True, "facecolor": "#000000", "edgecolor": "#000000", "linewidth": 0.1, "alpha": 1.0, "simplify_tolerance": 0.00005, "min_size_threshold": 0},
        "buildings": {"enabled": True, "facecolor": "#000000", "edgecolor": "#000000", "linewidth": 0.2, "alpha": 1.0, "simplify_tolerance": 0.000001, "hatch": "|", "zorder": 2, "size_categories": [], "min_size_threshold": 10},
        "water": {"enabled": True, "facecolor": "#000000", "edgecolor": "#000000", "linewidth": 0.3, "alpha": 1.0, "simplify_tolerance": 0.0001, "hatch": "\\", "zorder": 1, "min_size_threshold": 0.000001}
    },
    "output": {"separate_layers": False, "filename_prefix": "test_map", "output_directory": "output", "figure_size": [10, 10], "background_color": "white", "figure_dpi": 300, "margin": 0.05},
    "processing": {"street_filter": []}
}

# Attempt to render the template
try:
    rendered_html = template.render(style=dummy_style, generated_image_path=None)
    print("Template rendered successfully!")
    # Optionally, save the rendered HTML to a file for inspection
    # with open("rendered_test.html", "w") as f:
    #     f.write(rendered_html)
except Exception as e:
    print(f"Error rendering template: {e}")

