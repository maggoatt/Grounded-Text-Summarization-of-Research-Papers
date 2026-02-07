
import json

notebook_path = './Data_Pipeline.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook_data = json.load(f)

# The new source code
new_source = [
    "import urllib.request\n",
    "import gzip\n",
    "import json\n",
    "\n",
    "# Get the first file URL from the response dataset dynamically\n",
    "if 'response_dataset' in locals():\n",
    "    # You can change [0] to [-1] or any index you prefer\n",
    "    files = response_dataset.json().get('files', [])\n",
    "    if files:\n",
    "        url = files[0]\n",
    "        print(f\"Processing URL: {url}\")\n",
    "\n",
    "        print(\"Downloading and streaming first document...\")\n",
    "        try:\n",
    "            with urllib.request.urlopen(url) as response:\n",
    "                with gzip.GzipFile(fileobj=response) as gz:\n",
    "                    # Read the first line (first paper)\n",
    "                    first_line = gz.readline().decode('utf-8')\n",
    "                    \n",
    "                    if first_line:\n",
    "                        data = json.loads(first_line)\n",
    "                        \n",
    "                        # 1. Print keys to see structure\n",
    "                        print(\"\\n--- Top Level Keys ---\")\n",
    "                        print(list(data.keys()))\n",
    "                        \n",
    "                        if 'content' in data:\n",
    "                            print(\"\\n--- Content Keys ---\")\n",
    "                            print(list(data['content'].keys()))\n",
    "                            \n",
    "                            # 2. Get the full text\n",
    "                            full_text = data['content'].get('text', 'No text field found')\n",
    "                            \n",
    "                            # Print a much larger chunk, e.g. 5000 characters\n",
    "                            print(f\"\\n--- Full Text Preview (First 5000 chars of {len(full_text)} total) ---\")\n",
    "                            print(full_text[:5000] + \"... [truncated]\")\n",
    "                        else:\n",
    "                            print(\"\\nNo 'content' field found in data.\")\n",
    "                            print(json.dumps(data, indent=2)[:1000])\n",
    "                            \n",
    "                    else:\n",
    "                        print(\"File is empty.\")\n",
    "        except Exception as e:\n",
    "            print(f\"Failed: {e}\")\n",
    "    else:\n",
    "        print(\"No files found in response_dataset.\")\n",
    "else:\n",
    "    print(\"response_dataset variable not found. Please run the previous cells to fetch the dataset info.\")"
]

# Update the last cell
notebook_data['cells'][-1]['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_data, f, indent=1)

print("Notebook updated successfully.")
