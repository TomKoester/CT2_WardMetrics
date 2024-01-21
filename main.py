import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from wardmetrics.core_methods import eval_events, eval_segments
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from wardmetrics.visualisations import *
from wardmetrics.utils import *
import wardmetrics


def read_in(ctm_file_path):
    with open(ctm_file_path, 'r') as file:
        lines = file.readlines()

    lines = lines[1:]

    data = []

    for line_number, line in enumerate(lines, start=1):  # Start line numbering from 1
        columns = line.strip().split(',')

        timestamp = float(columns[0])
        label = columns[1]
        accelerometer_x = float(columns[2])
        accelerometer_y = float(columns[3])
        accelerometer_z = float(columns[4])
        gyro_x = float(columns[5])
        gyro_y = float(columns[6])
        gyro_z = float(columns[7])
        linear_accel_x = float(columns[8])
        linear_accel_y = float(columns[9])
        linear_accel_z = float(columns[10])
        gravity_x = float(columns[11])
        gravity_y = float(columns[12])
        gravity_z = float(columns[13])
        orientation_x = float(columns[14])
        orientation_y = float(columns[15])
        orientation_z = float(columns[16])
        rotation_vector_x = float(columns[17])
        rotation_vector_y = float(columns[18])
        rotation_vector_z = float(columns[19])
        rotation_vector_scalar = float(columns[20])
        rotation_vector_heading_accuracy = float(columns[21])
        magnetic_field_x = float(columns[22])
        magnetic_field_y = float(columns[23])
        magnetic_field_z = float(columns[24])

        # If nessasary, we can refactor this (column[0] instead of timestamp,...) but
        # i think at the moment this is better for visiblity
        data.append([timestamp, label, accelerometer_x, accelerometer_y, accelerometer_z, gyro_x, gyro_y, gyro_z,
                     linear_accel_x, linear_accel_y, linear_accel_z, gravity_x, gravity_y, gravity_z,
                     orientation_x, orientation_y, orientation_z, rotation_vector_x, rotation_vector_y,
                     rotation_vector_z, rotation_vector_scalar, rotation_vector_heading_accuracy, magnetic_field_x,
                     magnetic_field_y, magnetic_field_z])

    columns = [
        'timestamp', 'label',
        'accelerometer_x', 'accelerometer_y', 'accelerometer_z',
        'gyro_x', 'gyro_y', 'gyro_z',
        'linear_accel_x', 'linear_accel_y', 'linear_accel_z',
        'gravity_x', 'gravity_y', 'gravity_z',
        'orientation_x', 'orientation_y', 'orientation_z',
        'rotation_vector_x', 'rotation_vector_y', 'rotation_vector_z',
        'rotation_vector_scalar', 'rotation_vector_heading_accuracy',
        'magnetic_field_x', 'magnetic_field_y', 'magnetic_field_z'
    ]

    df = pd.DataFrame(data, columns=columns)

    df = df.dropna()

    df.index = df.index + 1
    df.index.name = 'Line Number'

    return df


def preprocessing(df):
    label_column = df['label']
    data_columns = df.drop('label', axis=1)
    # test with linear acceleration
    data_columns = data_columns.drop(['rotation_vector_x', 'rotation_vector_y', 'rotation_vector_z',
                                      'linear_accel_x', 'linear_accel_y', 'linear_accel_z',
                                      'gravity_x', 'gravity_y', 'gravity_z', 'timestamp',
                                      'orientation_x', 'orientation_y', 'orientation_z',
                                      'rotation_vector_scalar', 'rotation_vector_heading_accuracy',
                                      'magnetic_field_x', 'magnetic_field_y', 'magnetic_field_z',

                                      ], axis=1)

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit and transform the data columns
    data_columns_scaled = scaler.fit_transform(data_columns)

    # Create a new DataFrame with the scaled features and the original label column
    df_scaled = pd.DataFrame(data_columns_scaled, columns=data_columns.columns)
    df_scaled['label'] = label_column
    df = df_scaled

    # Windowing
    window_size = 50
    overlap = 15

    windows = []
    labels = []

    for i in range(0, len(df) - window_size + 1, window_size - overlap):
        window_data = df.iloc[i:i + window_size]

        window_features = extract_features(window_data)

        windows.append(window_features)

        window_label = window_data['label'].mode().values[0]
        labels.append(window_label)

    feature_windows = np.array(windows)
    window_labels = np.array(labels)

    # feature_windows = np.delete(feature_windows, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  ], axis=1)

    return feature_windows, window_labels


def remove_zero_range(list_of_tuples):
    solution = []
    for curr_tuple in list_of_tuples:
        if curr_tuple[0] != curr_tuple[1]:
            solution.append(curr_tuple)
    return solution


def get_event(events, label):
    result = []
    for event in events:
        if event[2] == label:
            result.append(event)
    return result


def evaluate_segment_event_based(y_true, y_pred):
    gt_events = convert_to_events(y_true)
    pred_events = convert_to_events(y_pred)
    start, end, label = gt_events[-1]
    # print(gt_events)
    # returns a list without labels [(start:0 , end: 60),...

    gt_event_0 = get_event(gt_events, 0)
    gt_event_1 = get_event(gt_events, 1)
    gt_event_2 = get_event(gt_events, 2)
    gt_event_3 = get_event(gt_events, 3)
    gt_event_4 = get_event(gt_events, 4)

    pd_event_0 = get_event(pred_events, 0)
    pd_event_1 = get_event(pred_events, 1)
    pd_event_2 = get_event(pred_events, 2)
    pd_event_3 = get_event(pred_events, 3)
    pd_event_4 = get_event(pred_events, 4)

    print("\nlabel 0\n")
    calculate_event_score(gt_event_0, pd_event_0, end)
    print("\nlabel 1\n")
    calculate_event_score(gt_event_1, pd_event_1, end)
    print("\nlabel 2\n")
    calculate_event_score(gt_event_2, pd_event_2, end)
    print("\nlabel3\n")
    calculate_event_score(gt_event_3, pd_event_3, end)
    print("\nlabel4\n")
    calculate_event_score(gt_event_4, pd_event_4, end)
    print("\n")


def calculate_event_score(gt_event, pd_event, end):
    gt_events_without_label = [(x[0], x[1]) for x in gt_event]
    pd_events_without_label = [(x[0], x[1]) for x in pd_event]
    pd_events_without_zero_range = remove_zero_range(pd_events_without_label)
    gt_events_without_zero_range = remove_zero_range(gt_events_without_label)
    print(gt_events_without_label)
    print(pd_events_without_label)
    # without zero range
    print("\n wiothout zero range")
    print(gt_events_without_zero_range)
    print(pd_events_without_zero_range)
    if len(gt_events_without_zero_range) == 0:
        print("No ground truth events")
    elif len(pd_events_without_zero_range) == 0:
        print("No predicted events")
    else:
        gt_event_scores, det_event_scores, detailed_scores, standard_scores = eval_events(
            ground_truth_events=gt_events_without_zero_range, detected_events=pd_events_without_zero_range)
        # Segments
        twoset_results, segments_with_scores, segment_counts, normed_segment_counts = eval_segments(
            gt_events_without_zero_range, pd_events_without_zero_range, evaluation_end=end)
        # here we can plot results if needed
        # plot_events_with_event_scores(gt_event_scores, det_event_scores, gt_events_without_zero_range, pred_events_without_zero_range, show=False)
        # plot_event_analysis_diagram(detailed_scores, fontsize=8, use_percentage=True)
        print("\nEventBased:\n")
        print(gt_event_scores)
        print(det_event_scores)
        print(detailed_scores)
        print(standard_scores)
        print("\nSegmentBased:\n")
        print(twoset_results)
        print(segments_with_scores)
        print(segment_counts)
        print(normed_segment_counts)


# returns a list of label ranges like events[(start:0 , end: 60, label: 4),...]
def convert_to_events(labels):
    events = []
    current_event = None
    for i, label in enumerate(labels):
        if current_event is None:
            current_event = (i, i, label)
        elif label != current_event[2]:
            current_event = (current_event[0], i - 1, current_event[2])
            events.append(current_event)
            current_event = (i, i, label)

    if current_event is not None:
        current_event = (current_event[0], i, current_event[2])
        events.append(current_event)

    return events


# TODO check which features are unimportant and test correlation as feature
def extract_features(window_data):
    basic_features = window_data.drop('label', axis=1).agg(['mean', 'std', 'skew', 'max', 'min', 'median', 'var',
                                                            ]).values.flatten()
    return basic_features


def evaluate_performance(y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)
    class_accuracy = {}
    unique_classes = set(y_true)
    for label in unique_classes:
        indices = (y_true == label)
        class_accuracy[label] = accuracy_score(y_true[indices], y_pred[indices])

    recall = recall_score(y_true, y_pred, average=None)
    precision = precision_score(y_true, y_pred, average=None, )
    f1 = f1_score(y_true, y_pred, average=None)

    return class_accuracy, recall, precision, f1


def help(train, test, set):
    df_test = read_in(test)

    df_training = read_in(train)
    # counts = df_training['label'].value_counts()
    # print(counts)
    # labels = df_training['label'].unique()
    # colors = plt.cm.rainbow(np.linspace(0,1,len(labels)))

    # for i, label in enumerate(labels):
    #     df_label = df_training[df_training['label'] == label]
    #     plt.plot(df_label['timestamp'], df_label['gyro_x'], label = label, color=colors[i],marker='o', linestyle='-', markersize=1)
    # plt.xlabel('Timestamp')
    # plt.ylabel('Gyro x')
    # plt.legend()

    # plt.show()
    # print(df_test)
    # print(df_training)

    # Preprocess the data for training
    # Label Encoding like encodes 'walking' as '0' and 'standing' as '1' and so on...
    label_encoder = LabelEncoder()
    df_training['label'] = label_encoder.fit_transform(df_training['label'])
    df_test['label'] = label_encoder.transform(df_test['label'])

    # Preprocess the data for training
    X_train, y_train = preprocessing(df_training)

    # Preprocess the test data for evaluation
    X_test, y_true = preprocessing(df_test)

    dt_classifier = DecisionTreeClassifier(criterion="entropy", class_weight="balanced", max_depth=8,
                                           max_features='sqrt', min_samples_leaf=3, splitter="best",
                                           random_state=2

                                           )

    dt_classifier.fit(X_train, y_train)

    # plt.figure(figsize=(18, 12),dpi=300)
    # plot_tree(dt_classifier, filled=True, feature_names=[f'Feature {i}' for i in range(X_train.shape[1])],
    #         class_names=label_encoder.classes_)
    # plt.show()

    # Predictions
    y_pred = dt_classifier.predict(X_test)

    # Evaluate the model
    accuracy, recall, precision, f1 = evaluate_performance(y_true, y_pred)

    print("Traditional Metrics Set " + set + ":")
    for i in [0, 1, 2, 3, 4]:
        print(f"Label {i}:" )
        print(f"  Accuracy: {accuracy[i]:.2f}")
        print(f"  Recall: {recall[i]:.2f}")
        print(f"  Precision: {precision[i]:.2f}")
        print(f"  F1 Score: {f1[i]:.2f}")
    # print("\nWard Metrics Set " + set + ":")
    # evaluate_segment_event_based(y_true, y_pred)


def main():
    ctm_file_path_training1 = r"DATASET\DATASET\P-1_training.ctm"
    ctm_file_path_test1 = r"DATASET\DATASET\P-1_test.ctm"
    # Reads in the provided sample data (adjust the path if you want to run it)
    help(test=ctm_file_path_test1, train=ctm_file_path_training1, set="1")

    ctm_file_path_training2 = r"DATASET\DATASET\P-2_training.ctm"
    ctm_file_path_test2 = r"DATASET\DATASET\P-2_test.ctm"
    # Reads in the provided sample data (adjust the path if you want to run it)
    help(test=ctm_file_path_test2, train=ctm_file_path_training2, set="2")

    ctm_file_path_training3 = r"DATASET\DATASET\P-3_training.ctm"
    ctm_file_path_test3 = r"DATASET\DATASET\P-3_test.ctm"
    # Reads in the provided sample data (adjust the path if you want to run it)
    help(test=ctm_file_path_test3, train=ctm_file_path_training3, set="3")

    ctm_file_path_training4 = r"DATASET\DATASET\P-4_training.ctm"
    ctm_file_path_test4 = r"DATASET\DATASET\P-4_test.ctm"
    # Reads in the provided sample data (adjust the path if you want to run it)
    help(test=ctm_file_path_test4, train=ctm_file_path_training4, set="4")

    ctm_file_path_training5 = r"DATASET\DATASET\P-5_training.ctm"
    ctm_file_path_test5 = r"DATASET\DATASET\P-5_test.ctm"
    # Reads in the provided sample data (adjust the path if you want to run it)
    help(test=ctm_file_path_test5, train=ctm_file_path_training5, set="5")


if __name__ == "__main__":
    main()
