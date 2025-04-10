import json

def main():
    with open("Results/Kirkley.json", 'r') as f:
        kirkley = json.load(f)
    for system in kirkley:
        kirkley[system]["scale-free"] = []
    with open("Results/Hubs.json", 'w') as f:
        json.dump(kirkley, f, indent=4)

if __name__ == "__main__":
    main()