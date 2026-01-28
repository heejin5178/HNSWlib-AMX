#include <string>
using namespace std;

void sift_test1B();
// void sift_test1B(const string &config_path = "", const string &log_path = "");

int main(int argc, char *argv[]) {
    string config_path = (argc > 1) ? argv[1] : "";
    string log_path = (argc > 2) ? argv[2] : "";
    sift_test1B();
    // sift_test1B(config_path, log_path);

    return 0;
}
