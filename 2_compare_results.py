import json
import numpy as np

def load_json_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def average_forecasts(forecasts):
    if isinstance(forecasts[0]['forecast'], list):
        num_options = len(forecasts[0]['forecast'])
        avg_forecasts = np.zeros(num_options)
        for forecast in forecasts:
            avg_forecasts += np.array(forecast['forecast'])
        avg_forecasts /= len(forecasts)
    else:
        avg_forecasts = np.mean([forecast['forecast'] for forecast in forecasts])
    return avg_forecasts

def average_perplexity(perplexity_values):
    return [np.mean(values) for values in perplexity_values]

def is_binary_choice(choices):
    return set(choices) == {"yes", "no"}

def get_forecast_data(result_data, autocast_data):
    total_questions = len(result_data)
    correct_answers = 0
    correct_model_predictions = 0
    consistent_forecasts = 0

    correct_answers_tf = 0
    correct_model_predictions_tf = 0
    consistent_forecasts_tf = 0
    tf_questions_count = 0

    correct_answers_mcq = 0
    correct_model_predictions_mcq = 0
    consistent_forecasts_mcq = 0
    mcq_questions_count = 0

    for result in result_data:
        result_id = result['id']
        print(f"Processing question ID {result_id}...")

        autocast_question = next((q for q in autocast_data if q['id'] == result_id), None)
        if autocast_question is None:
            print(f"No corresponding question found in autocast data for ID {result_id}.")
            continue

        question_text = autocast_question['question']
        correct_answer = autocast_question['answer']
        choices = autocast_question['choices']

        if is_binary_choice(choices):
            if correct_answer == "yes":
                correct_answer_index = 0
            elif correct_answer == "no":
                correct_answer_index = 1
            else:
                print(f"Error: Correct answer for ID {result_id} is not 'yes' or 'no'.")
                continue
            tf_questions_count += 1
        else:
            try:
                correct_answer_index = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.index(correct_answer)
            except ValueError:
                print(f"Error: Correct answer for ID {result_id} is not a valid letter.")
                continue
            mcq_questions_count += 1

        correct_answer_text = choices[correct_answer_index]

        print(f"Question ID: {result_id}")
        print(f"Question Text: {question_text}")
        print(f"Correct Answer: {correct_answer_text}")
        print(f"Choices: {choices}")

        crowd_forecasts = autocast_question['crowd']
        avg_crowd_forecasts = average_forecasts(crowd_forecasts)
        if is_binary_choice(choices):
            people_forecast_index = 0 if avg_crowd_forecasts >= 0.5 else 1
        else:
            people_forecast_index = np.argmax(avg_crowd_forecasts)

        people_forecast_choice = choices[people_forecast_index]

        perplexity_values = result['perplexity_values']
        if len(perplexity_values) != len(choices):
            print(f"Mismatch between number of choices and perplexity values for ID {result_id}.")
            continue

        avg_perplexity_values = average_perplexity(perplexity_values)
        model_forecast_index = np.argmin(avg_perplexity_values)
        if model_forecast_index >= len(choices):
            print(f"Index out of range for choices in ID {result_id}. Skipping this question.")
            continue
        model_forecast_choice = choices[model_forecast_index]

        print(f"\tFor question ID {result_id}: \"{question_text}\"")
        print(f"\tThe correct answer is: {correct_answer} ({correct_answer_text})")
        print(
            f"\tForecast by people: {people_forecast_choice} with average probability {avg_crowd_forecasts if isinstance(avg_crowd_forecasts, float) else avg_crowd_forecasts[people_forecast_index]:.4f}")
        print(
            f"\tForecast by model: {model_forecast_choice} with average perplexity {avg_perplexity_values[model_forecast_index]:.4f}\n")

        if people_forecast_index == correct_answer_index:
            correct_answers += 1
            if is_binary_choice(choices):
                correct_answers_tf += 1
            else:
                correct_answers_mcq += 1
        if model_forecast_index == correct_answer_index:
            correct_model_predictions += 1
            if is_binary_choice(choices):
                correct_model_predictions_tf += 1
            else:
                correct_model_predictions_mcq += 1
        if people_forecast_choice == model_forecast_choice:
            consistent_forecasts += 1
            if is_binary_choice(choices):
                consistent_forecasts_tf += 1
            else:
                consistent_forecasts_mcq += 1

    people_accuracy = (correct_answers / total_questions) * 100
    model_accuracy = (correct_model_predictions / total_questions) * 100
    consistency_percentage = (consistent_forecasts / total_questions) * 100

    if tf_questions_count > 0:
        people_accuracy_tf = (correct_answers_tf / tf_questions_count) * 100
        model_accuracy_tf = (correct_model_predictions_tf / tf_questions_count) * 100
        consistency_percentage_tf = (consistent_forecasts_tf / tf_questions_count) * 100

    if mcq_questions_count > 0:
        people_accuracy_mcq = (correct_answers_mcq / mcq_questions_count) * 100
        model_accuracy_mcq = (correct_model_predictions_mcq / mcq_questions_count) * 100
        consistency_percentage_mcq = (consistent_forecasts_mcq / mcq_questions_count) * 100

    print(
        f"People's forecasts were correct for {correct_answers}/{total_questions} questions ({people_accuracy:.2f}%).")
    print(
        f"Model's forecasts were correct for {correct_model_predictions}/{total_questions} questions ({model_accuracy:.2f}%).")
    print(
        f"The forecasts of people and the model matched for {consistent_forecasts}/{total_questions} questions ({consistency_percentage:.2f}%).")

    if tf_questions_count > 0:
        print(f"\nTrue/False Questions: {tf_questions_count} total")
        print(
            f"People's forecasts were correct for {correct_answers_tf}/{tf_questions_count} questions ({people_accuracy_tf:.2f}%).")
        print(
            f"Model's forecasts were correct for {correct_model_predictions_tf}/{tf_questions_count} questions ({model_accuracy_tf:.2f}%).")
        print(
            f"The forecasts of people and the model matched for {consistent_forecasts_tf}/{tf_questions_count} questions ({consistency_percentage_tf:.2f}%).")

    if mcq_questions_count > 0:
        print(f"\nMultiple-Choice Questions (MCQ) with more than two options: {mcq_questions_count} total")
        print(
            f"People's forecasts were correct for {correct_answers_mcq}/{mcq_questions_count} questions ({people_accuracy_mcq:.2f}%).")
        print(
            f"Model's forecasts were correct for {correct_model_predictions_mcq}/{mcq_questions_count} questions ({model_accuracy_mcq:.2f}%).")
        print(
            f"The forecasts of people and the model matched for {consistent_forecasts_mcq}/{mcq_questions_count} questions ({consistency_percentage_mcq:.2f}%).")

def main():
    result_filepath = 'result.json'
    autocast_filepath = 'autocast_adjusted.json'

    print("Loading JSON data...")
    result_data = load_json_data(result_filepath)
    autocast_data = load_json_data(autocast_filepath)

    print("Processing forecasts and calculating accuracy...")
    get_forecast_data(result_data, autocast_data)

if __name__ == "__main__":
    main()
