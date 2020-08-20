#pragma once

#include "network_2d.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class ExtensibleNetwork2D : public Network2D<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<ExtensibleNetwork2D<T>>;

        ExtensibleNetwork2D(const size_t inputWidthInFirstLayer,
                            const size_t inputHeightInFirstLayer,
                            const size_t inputCountInFirstLayer,

                            const size_t filterWidthInFirstLayer,
                            const size_t filterHeightInFirstLayer,
                            const size_t filterCountInFirstLayer);

        void PushBack(const size_t stepSizeInNewLayer);

        void PushBack(const size_t filterWidthInNewLayer,
                      const size_t filterHeightInNewLayer,
                      const size_t filterCountInNewLayer);

      };

      template <typename T>
      ExtensibleNetwork2D<T>::ExtensibleNetwork2D(const size_t inputWidthInFirstLayer,
                                                  const size_t inputHeightInFirstLayer,
                                                  const size_t inputCountInFirstLayer,

                                                  const size_t filterWidthInFirstLayer,
                                                  const size_t filterHeightInFirstLayer,
                                                  const size_t filterCountInFirstLayer)
        :
        Network2D<T>{ inputWidthInFirstLayer,
                      inputHeightInFirstLayer,
                      inputCountInFirstLayer,

                      filterWidthInFirstLayer,
                      filterHeightInFirstLayer,
                      filterCountInFirstLayer }
      {
      }

      template <typename T>
      void ExtensibleNetwork2D<T>::PushBack(const size_t stepSizeInNewLayer)
      {
        Network2D<T>::PushBack(stepSizeInNewLayer);
      }

      template <typename T>
      void ExtensibleNetwork2D<T>::PushBack(const size_t filterWidthInNewLayer,
                                            const size_t filterHeightInNewLayer,
                                            const size_t filterCountInNewLayer)
      {
        Network2D<T>::PushBack(filterWidthInNewLayer,
                               filterHeightInNewLayer,
                               filterCountInNewLayer);
      }

    }
  }
}