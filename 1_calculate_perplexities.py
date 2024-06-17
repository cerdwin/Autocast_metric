import json
from perplexity_calculation import calculate_perplexity_for_time_window

def main():
    print("Starting perplexity calculation script.")

    json_input_path = '/home/zanzajen/training/davids/nanoGPT/autocast_adjusted.json'
    json_output_path = '/home/zanzajen/training/davids/nanoGPT/june_output.json'
    root_directory = '/home/zanzajen/training/davids/nanoGPT'

    try:
        print(f"Loading JSON data from {json_input_path}")
        with open(json_input_path, 'r') as file:
            data = json.load(file)

        print("Filtering questions with all time windows from 2020 or later.")
        filtered_data = [
            item for item in data 
            if all(timestamp >= "2020-01" for timestamp in item['time_window'])
        ]
        #filtered_data = filtered_data[:30]

        results = []

        for item in filtered_data:
            question_id = item['id']
            model_inputs = item['model_input']
            time_windows = item['time_window']
            print(f"Processing question ID: {question_id} with time windows: {time_windows}")

            sentence_perplexities = []
            for sentence in model_inputs:
                try:
                    print(f"Calculating perplexity for sentence: {sentence}")
                    perplexities = calculate_perplexity_for_time_window(root_directory, sentence, time_windows)
                    sentence_perplexities.append(perplexities)
                except Exception as e:
                    print(f"Error calculating perplexity for sentence: {sentence} - {e}")
                    sentence_perplexities.append([None] * len(time_windows))

            results.append({
                "id": question_id,
                "model_input": model_inputs,
                "time_windows": time_windows,
                "perplexity_values": sentence_perplexities
            })

        print(f"Saving results to {json_output_path}")
        with open(json_output_path, 'w') as output_file:
            json.dump(results, output_file, indent=4)

        print("Perplexity calculation script completed successfully.")

    except Exception as e:
        print(f"An error occurred during the process: {e}")

if __name__ == "__main__":
    main()
