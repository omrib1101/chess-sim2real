import pandas as pd
import os
import torch
from tqdm import tqdm
from ..test_utils.image_to_fen import get_fen_from_image

def expand_fen(fen):
    """
    Expands a compressed FEN string into a flat list of 64 characters.
    '/' are ignored, and digits (e.g., '8') are expanded into '.' (empty squares).
    """
    board_part = fen.split(' ')[0]
    expanded = []
    for char in board_part:
        if char == '/': 
            continue
        if char.isdigit():
            # Expand digit 'n' into n dots
            expanded.extend(['.'] * int(char))
        else:
            expanded.append(char)
    return expanded

# 1. Prediction Generation Function
def generate_comparison_csv(image_folder, ground_truth_csv, model_path, output_path, predict_fen_with_model):
    """
    Applies the model to every image in the test folder.
    Saves a CSV containing: filename, true_fen, and pred_fen.
    """
    df_gt = pd.read_csv(ground_truth_csv)
    results = []
    
    print(f"Generating predictions for folder: {image_folder}")
    
    # Iterate through ground truth CSV entries
    for _, row in tqdm(df_gt.iterrows(), total=len(df_gt)):
        # Format frame number to match file naming (e.g., 200 -> '000200')
        frame_num = str(row['to_frame']).zfill(6)
        
        # Look for any file in the folder containing the frame number string
        img_name_list = [f for f in os.listdir(image_folder) if f"frame_{frame_num}" in f]
        
        if not img_name_list:
            continue
            
        img_path = os.path.join(image_folder, img_name_list[0])
        
        # Run the model using the provided inference function
        pred_fen = predict_fen_with_model(img_path, model_path)
        
        results.append({
            'filename': img_name_list[0],
            'true_fen': row['fen'],
            'pred_fen': pred_fen
        })
    
    # Save the accumulated results to a new CSV
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    return output_path

# 2. Overall Accuracy Calculation
def calculate_overall_accuracy(csv_path):
    """
    Calculates the total accuracy across all 64 squares of all images.
    Returns: correct count, total count, and percentage.
    """
    df = pd.read_csv(csv_path)
    total_squares = 0
    correct_squares = 0
    
    for _, row in df.iterrows():
        true_board = expand_fen(row['true_fen'])
        pred_board = expand_fen(row['pred_fen'])
        
        total_squares += 64
        # Compare square by square
        correct_squares += sum(1 for t, p in zip(true_board, pred_board) if t == p)
        
    accuracy = (correct_squares / total_squares) * 100 if total_squares > 0 else 0
    return correct_squares, total_squares, accuracy

# 3. Class-Specific Accuracy
def calculate_class_accuracy(csv_path):
    """
    Calculates accuracy for each of the 13 classes (P, R, N, B, Q, K, p, r, n, b, q, k, .).
    Example: Out of all real Black Kings, how many were predicted as Black Kings.
    """
    df = pd.read_csv(csv_path)
    classes = ['P', 'R', 'N', 'B', 'Q', 'K', 'p', 'r', 'n', 'b', 'q', 'k', '.']
    class_stats = {c: {'correct': 0, 'total': 0} for c in classes}
    
    for _, row in df.iterrows():
        true_board = expand_fen(row['true_fen'])
        pred_board = expand_fen(row['pred_fen'])
        
        for t, p in zip(true_board, pred_board):
            class_stats[t]['total'] += 1
            if t == p:
                class_stats[t]['correct'] += 1
                
    result_dict = {}
    for c, stats in class_stats.items():
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        result_dict[c] = acc
    return result_dict

# 4. Color Detection Accuracy (Conditional on Piece Existence)
def calculate_color_accuracy(csv_path):
    """
    Focuses only on squares that contain a piece in the Ground Truth.
    Checks if the model predicted the correct color (Black/White), regardless of piece type.
    Predicting 'empty' for a piece is counted as an error.
    """
    df = pd.read_csv(csv_path)
    color_stats = {'black': {'correct': 0, 'total': 0}, 'white': {'correct': 0, 'total': 0}}
    
    for _, row in df.iterrows():
        true_board = expand_fen(row['true_fen'])
        pred_board = expand_fen(row['pred_fen'])
        
        for t, p in zip(true_board, pred_board):
            # Ignore empty squares in Ground Truth as per instructions
            if t == '.': 
                continue 
            
            # Identify Ground Truth color: Uppercase is White, Lowercase is Black
            true_color = 'white' if t.isupper() else 'black'
            color_stats[true_color]['total'] += 1
            
            # If model predicts a piece (not empty)
            if p != '.':
                pred_color = 'white' if p.isupper() else 'black'
                if true_color == pred_color:
                    color_stats[true_color]['correct'] += 1
            # Note: if p == '.', correct count is not incremented (counted as error)
                    
    results = {
        'black_color_acc': (color_stats['black']['correct'] / color_stats['black']['total'] * 100) if color_stats['black']['total'] > 0 else 0,
        'white_color_acc': (color_stats['white']['correct'] / color_stats['white']['total'] * 100) if color_stats['white']['total'] > 0 else 0
    }
    return results

# 5. Master Aggregator Function
def run_full_statistics_report(image_folder, gt_csv, model_path, dst_folder, predict_fen_with_model):
    """
    Orchestrator: Generates predictions, calculates all metrics, 
    and saves a consolidated summary CSV.
    """
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
        
    comparison_csv = os.path.join(dst_folder, "model_predictions_comparison.csv")
    
    # Step 1: Generate Predictions
    generate_comparison_csv(image_folder, gt_csv, model_path, comparison_csv, predict_fen_with_model)
    
    # Step 2-4: Run analysis functions
    correct, total, overall_acc = calculate_overall_accuracy(comparison_csv)
    class_accs = calculate_class_accuracy(comparison_csv)
    color_accs = calculate_color_accuracy(comparison_csv)
    
    # Step 5: Consolidate data into a Summary Dictionary for a Final Report
    summary_data = {
        'Metric': ['Overall_Accuracy', 'Total_Squares_Tested', 'Total_Correct_Squares'],
        'Value': [overall_acc, total, correct]
    }
    
    # Add piece-wise accuracy
    for piece, acc in class_accs.items():
        summary_data['Metric'].append(f'Piece_Accuracy_{piece}')
        summary_data['Value'].append(acc)
        
    # Add color accuracy
    for color, acc in color_accs.items():
        summary_data['Metric'].append(f'Color_Accuracy_{color}')
        summary_data['Value'].append(acc)
        
    # Save the consolidated report
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(dst_folder, "final_statistics_report.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print(f"Full analysis complete. Consolidated report saved at: {summary_path}")
    return summary_df

run_full_statistics_report(r"/home/noareg/my_project/data_for_test/test/images", r"/home/noareg/my_project/data_for_test/test/test.csv", 
                           r"/home/noareg/my_project/codes/model/combined_data_model.pth", r"/home/noareg/my_project/statistics_results/model3.2",
                           get_fen_from_image)