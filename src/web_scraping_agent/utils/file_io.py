from datetime import datetime
from pathlib import Path


def save_markdown(company_name, task_output):
    # Get today's date in the format YYYY-MM-DD
    now_time = datetime.now().strftime('%Y%m%d%H%M%S')
    # Set the filename with today's date
    output_path = Path(__file__).parents[1] / "output"
    output_path.mkdir(exist_ok=True)
    filename = f"{company_name}-{now_time}.md"
    # Write the task output to the markdown file
    with output_path.joinpath(filename).open("w+") as file:
        file.write(task_output.raw_output)
    print(f"Newsletter saved as {filename}")
