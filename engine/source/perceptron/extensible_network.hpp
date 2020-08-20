#pragma once

#include "network.hpp"

namespace cnn
{
  namespace engine
  {
    namespace perceptron
    {
      template <typename T>
      class ExtensibleNetwork : public Network<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<ExtensibleNetwork<T>>;

        ExtensibleNetwork(const size_t inputCountInFirstLayer,
                          const size_t outputCountInFirstLayer);

        void PushBack(const size_t outputCountInNewLayer);

      };

      template <typename T>
      ExtensibleNetwork<T>::ExtensibleNetwork(const size_t inputCountInFirstLayer,
                                              const size_t outputCountInFirstLayer)
        :
        Network<T>{ inputCountInFirstLayer,
                    outputCountInFirstLayer }
      {
      }

      template <typename T>
      void ExtensibleNetwork<T>::PushBack(const size_t outputCountInNewLayer)
      {
        Network<T>::PushBack(outputCountInNewLayer);
      }
    }
  }
}