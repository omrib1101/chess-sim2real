synthetic_data_generator/data_generator_from_csv_files.py - to generate chess images from FENs in a CSV file

synthetic_data_generator/data_generator_random.py - to generate random chess images with controlled density and view distribution

images_crop/crop_synthetic.py - to crop chess images returned from the Blender to focus on the board area

images_crop/shift_and_pad.py - to shift and pad real chess images for data augmentation (only used for training data)

images_crop/split_to_squares.py - to split chess board images into individual square images with filenames indicating frame number, view, count, and label (we use it for the training data)

prep_for_train/split_train_val.py - to split the prepared square images into training and validation sets organized by piece type

models/train_zero_shot_model.py - to train the zero-shot model

models/train_fine_tuned_model.py - to fine-tune the zero-shot model

models/train_combined_data_model.py - to train the fine-tuned model on combined synthetic and real data

test_utils/image_to_fen.py - get_fen_from_imagefunction, that given an image path and a model path, returns the predicted FEN string

test_utils/fen_to_board.py - fen_to_board_image function, that given a FEN string, output path, and a view (white or black), generates a chessboard image representing the FEN 

statistics/stats_utils.py - to run the statistics on the model predictions vs. ground truth FENs. 
Contains an orchestrating function run_full_statistics_report that given a path to test images, grount truth CSV file, and model path, runs the full statistics report and saves it to a CSV file.
