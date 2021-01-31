#include "Builder.hpp"

// TODO: We must add smart and fast autofit for topologies (It's very important for convenience!).

int main()
{
  auto library = cnn::learning_example::Builder<float>::GetLessonLibrary();
  auto network = cnn::learning_example::Builder<float>::GetNetwork();

  return 0;
}

