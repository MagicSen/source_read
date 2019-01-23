#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <glog/logging.h>

#include <cstring>
#include <cstdlib>
#include <set>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/mpi_templates.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;
using caffe::caffe_scal;


DEFINE_string(gpu, "",
    "Run in GPU mode on given comma-separated device IDs.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning. "
    "Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

static std::vector<int> GetDevicesFromFlag() {
  std::vector<int> gpus;
  if (FLAGS_gpu != "") {
    std::vector<std::string> gpus_str;
    boost::split(gpus_str, FLAGS_gpu, boost::is_any_of(","));
    for (int i = 0; i < gpus_str.size(); ++i) {
      gpus.push_back(atoi(gpus_str[i].c_str()));
    }
  }
  return gpus;
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  std::vector<int> gpus = GetDevicesFromFlag();
  CHECK_GT(gpus.size(), 0) << "Need at least one device ID to query.";
  for (int i = 0; i < gpus.size(); ++i) {
    LOG(INFO) << "Querying device ID = " << gpus[i];
    caffe::Caffe::SetDevice(gpus[i]);
    caffe::Caffe::DeviceQuery();
  }
  return 0;
}
RegisterBrewFunction(device_query);

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(",") );
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver->net()->CopyTrainedLayersFrom(model_names[i]);
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
    }
  }
}

// Train / Finetune a model.
int train() {
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";

  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(FLAGS_solver, &solver_param);

  // If the gpu flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  std::vector<int> gpus = GetDevicesFromFlag();
  if (gpus.size() == 0
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
    if (solver_param.device_id_size() > 0) {
      for (int i = 0; i < solver_param.device_id_size(); ++i) {
        gpus.push_back(solver_param.device_id(i));
      }
    } else {
      gpus.push_back(0);
    }
  }

  int gpu_id = gpus.size() == 0 ? -1 : gpus[0];
#ifdef USE_MPI
  // Check whether the number of MPI processors matches the number of devices.
  if (gpus.size() > 0) {
    if (Caffe::mpi_rank() == 0) {
      CHECK_EQ(Caffe::mpi_size(), gpus.size())
          << "The number of MPI processors should match"
             "the number of GPU devices provided";
    }
    gpu_id = gpus[Caffe::mpi_rank()];
  }
#endif

  // Set device id and mode
  if (gpu_id >= 0) {
    LOG(INFO) << "Use GPU with device ID " << gpu_id;
    Caffe::SetDevice(gpu_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  LOG(INFO) << "Starting Optimization";
  shared_ptr<caffe::Solver<float> >
    solver(caffe::GetSolver<float>(solver_param));

  if (FLAGS_snapshot.size()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver->Solve(FLAGS_snapshot);
  } else if (FLAGS_weights.size()) {
    CopyLayers(&*solver, FLAGS_weights);
    solver->Solve();
  } else {
    solver->Solve();
  }
  LOG(INFO) << "Optimization Done.";
  return 0;
}
RegisterBrewFunction(train);


// Test: score a model.
int test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

  std::vector<int> gpus = GetDevicesFromFlag();
  int gpu_id = gpus.size() == 0 ? -1 : gpus[0];
#ifdef USE_MPI
  // Check whether the number of MPI processors matches the number of devices.
  if (gpus.size() > 0) {
    if (Caffe::mpi_rank() == 0) {
      CHECK_EQ(Caffe::mpi_size(), gpus.size())
          << "The number of MPI processors should match"
             "the number of GPU devices provided";
    }
    gpu_id = gpus[Caffe::mpi_rank()];
  }
#endif

  // Set device id and mode
  if (gpu_id >= 0) {
    LOG(INFO) << "Use GPU with device ID " << gpu_id;
    Caffe::SetDevice(gpu_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    float iter_loss;
    const vector<Blob<float>*>& result =
        caffe_net.Forward(bottom_vec, &iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net.blob_loss_weights()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }

  return 0;
}
RegisterBrewFunction(test);


// Time: benchmark the execution time of a model.
int time() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";

  std::vector<int> gpus = GetDevicesFromFlag();
  int gpu_id = gpus.size() == 0 ? -1 : gpus[0];
#ifdef USE_MPI
  // Check whether the number of MPI processors matches the number of devices.
  if (gpus.size() > 0) {
    if (Caffe::mpi_rank() == 0) {
      CHECK_EQ(Caffe::mpi_size(), gpus.size())
          << "The number of MPI processors should match"
             "the number of GPU devices provided";
    }
    gpu_id = gpus[Caffe::mpi_rank()];
  }
#endif

  // Set device id and mode
  if (gpu_id >= 0) {
    LOG(INFO) << "Use GPU with device ID " << gpu_id;
    Caffe::SetDevice(gpu_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TRAIN);

  // Do a clean forward and backward pass, so that memory allocation are done
  // and future iterations will be more stable.
  LOG(INFO) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  float initial_loss;
  caffe_net.Forward(vector<Blob<float>*>(), &initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;
  LOG(INFO) << "Performing Backward";
  caffe_net.Backward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
      caffe_net.bottom_need_backward();
  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  Timer backward_timer;
  Timer comm_timer;
  Timer timer;
  std::vector<double> forward_time_per_layer(layers.size(), 0.0);
  std::vector<double> backward_time_per_layer(layers.size(), 0.0);
  std::vector<double> comm_time_per_layer(layers.size(), 0.0);
  double forward_time = 0.0;
  double backward_time = 0.0;
#ifdef USE_MPI
  const std::set<std::string>& serial_layers = caffe_net.serial_layers();
  double comm_time = 0.0;
#endif
  for (int j = 0; j < FLAGS_iterations; ++j) {
    Timer iter_timer;
    iter_timer.Start();
    forward_timer.Start();
    for (int i = 0; i < layers.size(); ++i) {
      timer.Start();
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      forward_time_per_layer[i] += timer.MicroSeconds();
    }
    forward_time += forward_timer.MicroSeconds();
    backward_timer.Start();
    for (int i = layers.size() - 1; i >= 0; --i) {
      timer.Start();
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          bottom_vecs[i]);
      backward_time_per_layer[i] += timer.MicroSeconds();
    }
    backward_time += backward_timer.MicroSeconds();
#ifdef USE_MPI
    if (Caffe::mpi_size() > 1) {
      comm_timer.Start();
      for (int i = layers.size() - 1; i >= 0; --i) {
        if (serial_layers.find(layers[i]->layer_param().name()) !=
            serial_layers.end()) {
          comm_time_per_layer[i] = 0;
          continue;
        }
        const vector<shared_ptr<Blob<float> > >& blobs = layers[i]->blobs();
        timer.Start();
        for (int j = 0; j < blobs.size(); ++j) {
          if (Caffe::mpi_size() == 1) continue;
          MPIAllreduce<float>(blobs[j]->count(), MPI_IN_PLACE,
              blobs[j]->mutable_cpu_diff(), MPI_SUM);
          caffe_scal(blobs[j]->count(), 1.0f / Caffe::mpi_size(),
              blobs[j]->mutable_cpu_diff());
        }
        comm_time_per_layer[i] += timer.MicroSeconds();
      }
      comm_time += comm_timer.MicroSeconds();
    }
    LOG(INFO) << "Iteration: " << j + 1 << " forward-backward-comm time: "
      << iter_timer.MilliSeconds() << " ms.";
#else
    LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
      << iter_timer.MilliSeconds() << " ms.";
#endif
  }
  LOG(INFO) << "Average time per layer: ";
  for (int i = 0; i < layers.size(); ++i) {
    const caffe::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
      "\tforward: " << forward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
      "\tbackward: " << backward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
#ifdef USE_MPI
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
      "\tcomm: " << comm_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
#endif
  }
  total_timer.Stop();
  LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
    FLAGS_iterations << " ms.";
#ifdef USE_MPI
  LOG(INFO) << "Average Comm pass: " << comm_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Forward-Backward-Comm: " << total_timer.MilliSeconds() /
    FLAGS_iterations << " ms.";
#else
  LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
    FLAGS_iterations << " ms.";
#endif
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}
RegisterBrewFunction(time);

int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  train           train or finetune a model\n"
      "  test            score a model\n"
      "  device_query    show GPU diagnostic information\n"
      "  time            benchmark model execution time");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  if (argc == 2) {
#ifdef WITH_PYTHON_LAYER
    try {
#endif
      return GetBrewFunction(caffe::string(argv[1]))();
#ifdef WITH_PYTHON_LAYER
    } catch (bp::error_already_set) {
      PyErr_Print();
      return 1;
    }
#endif
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
  }
}
