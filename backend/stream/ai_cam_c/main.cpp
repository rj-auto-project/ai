#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
// #include <redis-cpp/stream.hpp> // Redis in C++
#include <hiredis/hiredis.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <ctime>
// #include <uuid/uuid.h> // For UUID generation

using namespace cv;
using namespace std;

// Initialize device and model
const int width = 640, height = 480;
string camera_ip = "198.78.45.89";
int camera_id = 2;
int fps = 30;

map<int, map<string, string>> custom_track_ids;
map<int, vector<string>> track_ids_inframe;
vector<string> known_track_ids;
torch::jit::script::Module model;

// string generate_custom_track_id(const string &label, float confidence) {
//     uuid_t uuid;
//     uuid_generate(uuid);
//     char uuid_str[37];
//     uuid_unparse_lower(uuid, uuid_str);
//     return label + "_" + to_string(confidence) + "_" + string(uuid_str);
// }

float calculate_pixel_distance(float x1, float y1, float x2, float y2) {
    return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

float calculate_speed(float pixel_distance, float time_interval) {
    return pixel_distance / time_interval;
}

void process_frame_batch(const vector<Mat> &frames, torch::Tensor &batch_results, vector<Mat> &resized_frames) {
    for (const auto &frame : frames) {
        Mat resized;
        resize(frame, resized, Size(width / 32 * 32, height / 32 * 32));
        resized_frames.push_back(resized);
    }

    torch::Tensor frames_tensor = torch::from_blob(resized_frames[0].data, {static_cast<int64_t>(frames.size()), 3, 640, 480}, torch::kUInt8);
    // torch::Tensor frames_tensor = torch::from_blob(resized_frames.data(), {frames.size(), 3, 640, 480}, torch::kFloat);
    frames_tensor = frames_tensor.permute({0, 3, 1, 2}).to(torch::kCUDA) / 255.0;

    torch::NoGradGuard no_grad;
    batch_results = model.forward({frames_tensor}).toTensor();
}

void track_objects(vector<Mat> &frames, const torch::Tensor &batch_results, double frame_time) {
    vector<int> current_track_ids;

    for (size_t i = 0; i < frames.size(); ++i) {
        Mat &frame = frames[i];

        // Process results for the current frame
        // Assuming batch_results is a tensor with bounding box and class info
        for (int j = 0; j < batch_results.size(0); ++j) {
            auto box = batch_results[i][j];

            int x1 = static_cast<int>(box[0].item<float>());
            int y1 = static_cast<int>(box[1].item<float>());
            int x2 = static_cast<int>(box[2].item<float>());
            int y2 = static_cast<int>(box[3].item<float>());
            float score = box[4].item<float>();
            string label = "some_label";  // Get the actual label from your model output

            int track_id = j; // Assuming the track_id is the index of the detection

            // Custom track ID
            // if (custom_track_ids.find(track_id) == custom_track_ids.end()) {
            //     string custom_id = generate_custom_track_id(label, score);
            //     custom_track_ids[track_id] = {
            //         {"custom_track_id", custom_id},
            //         {"camera_id", to_string(camera_id)},
            //         {"camera_ip", camera_ip},
            //         {"first_appearance", to_string(frame_time)},
            //         {"last_appearance", to_string(frame_time)}
            //     };
            // }

            // custom_track_ids[track_id]["last_appearance"] = to_string(frame_time);

            rectangle(frame, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2);
            putText(frame, custom_track_ids[track_id]["custom_track_id"], Point(x1, y1 - 10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 1);
        }
    }

    // Add logic to save track data when the object leaves the frame using Redis
    // (you need to use the Redis library to store data)
}

void stream_process(int camera_id, const string &camera_ip, const string &video_path, int batch_size) {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error opening video file: " << video_path << endl;
        return;
    }

    VideoWriter out("/home/annone/ai/data/output.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(width, height));

    vector<Mat> frames;
    double t1 = clock();

    while (cap.isOpened()) {
        Mat frame;
        if (!cap.read(frame)) break;

        double frame_time = clock();
        frames.push_back(frame);

        if (frames.size() >= batch_size) {
            torch::Tensor batch_results;
            vector<Mat> resized_frames;
            process_frame_batch(frames, batch_results, resized_frames);
            track_objects(resized_frames, batch_results, frame_time);

            for (const auto &tracked_frame : resized_frames) {
                // imshow("Tracked Frame", tracked_frame);
                out.write(tracked_frame);
            }

            frames.clear();
        }

        if (waitKey(1) == 'q') break;
    }

    double t2 = clock();
    cout << "Total time: " << (t2 - t1) / CLOCKS_PER_SEC << " seconds" << endl;

    cap.release();
    out.release();
    destroyAllWindows();
}

int main() {
    // Load the YOLO model (modify the path as needed)
    try {
        model = torch::jit::load("/home/annone/ai/models/yolov8n.torchscript");
        model.to(torch::kCUDA);
    } catch (const c10::Error &e) {
        cerr << "Error loading the YOLO model: " << e.what() << endl;
        return -1;
    }

    stream_process(camera_id, camera_ip, "/home/annone/ai/data/wrongway.mp4", 2);

    return 0;
}
