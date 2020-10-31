#include "backend.hpp"

#include <SFML/Graphics.hpp>

namespace cnn
{
  namespace examples
  {
    namespace complex_using
    {
      Backend::Backend()
        :
        Field_{ 640, 640 },
        Cursor_{ 32, 32 }
      {
        LoadField();
      }

      void Backend::LoadField()
      {
        sf::Image image;

        if (image.loadFromFile("...") == false)
        {
          throw std::runtime_error("cnn::examples::complex_using::Backend::LoadField(), Field_ could not be loaded from file.");
        }
        // ...
      }
    }
  }
}