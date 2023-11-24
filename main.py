import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import wardmetrics

def read_in(ctm_file_path):
    with open(ctm_file_path, 'r') as file:
        lines = file.readlines()

    lines = lines[1:]

    data = []

    # Process each line in the CTM file
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
        'timestamp', 'label', 'accelerometer_x', 'accelerometer_y', 'accelerometer_z', 'gyro_x', 'gyro_y', 'gyro_z',
        'linear_accel_x', 'linear_accel_y', 'linear_accel_z', 'gravity_x', 'gravity_y', 'gravity_z',
        'orientation_x', 'orientation_y', 'orientation_z', 'rotation_vector_x', 'rotation_vector_y',
        'rotation_vector_z', 'rotation_vector_scalar', 'rotation_vector_heading_accuracy', 'magnetic_field_x',
        'magnetic_field_y', 'magnetic_field_z'
    ]

    df = pd.DataFrame(data, columns=columns)

    # Set the line numbers as the index
    df.index = df.index + 1
    df.index.name = 'Line Number'


    return df


def preproccessing(df):
    #Windowing
    window_size = 100  # Adjust window size as needed
    overlap = 10  # Adjust overlap as needed
    windows = [df[i:i + window_size] for i in range(0, len(df), window_size - overlap)]


    #return feature_df
def evaluate_segment_event_based(y_true, y_pred):
    segment_metrics = wardmetrics.get_segment_metrics(y_true, y_pred)
    event_metrics = wardmetrics.get_event_metrics(y_true, y_pred)

    return segment_metrics, event_metrics
def evaluate_performance(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return accuracy, recall, precision, f1
def main():
    ctm_file_path_training =r"DATASET\DATASET\P-1_training.ctm"
    ctm_file_path_test = r"DATASET\DATASET\P-1_test.ctm"
    # Reads in the provided sample data (adjust the path if you want to run it)
    df_test = read_in(ctm_file_path_test)
    df_training = read_in(ctm_file_path_training)
    #print(df_test)
    #print(df_training)
    #preproccessed_df = preproccessing(df)
    #print(preproccessed_df)
    # Combine the training and test sets for label encoding


    combined_df = pd.concat([df_training, df_test], ignore_index=True)

    # Label Encoding like encodes 'walking' as '0' and 'standing' as '1' and so on...
    label_encoder = LabelEncoder()
    combined_df['label'] = label_encoder.fit_transform(combined_df['label'])

    df_training_encoded = combined_df.iloc[:len(df_training)]
    df_test_encoded = combined_df.iloc[len(df_training):]

    # Train-Test Split
    X_train, X_val, y_train, y_val = train_test_split(
        #entfernen von label column
        df_training_encoded.drop('label', axis=1),
        df_training_encoded['label'],
        test_size=0.2,
        random_state=42
    )

    # Decision Trees Classifier
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(X_train, y_train)

    # Predictions
    y_pred = dt_classifier.predict(df_test_encoded.drop('label', axis=1))


    accuracy, recall, precision, f1 = evaluate_performance(df_test_encoded['label'], y_pred)
    print("Traditional Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Evaluate using segment-based and event-based metrics
    segment_metrics, event_metrics = evaluate_segment_event_based(df_test_encoded['label'], y_pred)
    print("\nSegment-Based Metrics:")
    print(segment_metrics)
    print("\nEvent-Based Metrics:")
    print(event_metrics)

if __name__ == "__main__":
    main()
