#include "Builder.hpp"

int main()
{
  auto library = cnn::learning_example::Builder<float>::GetLessonLibrary();

  return 0;
}

