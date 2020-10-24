#include "SFML/Graphics.hpp"

int main(int argc, char** argv)
{
  sf::RenderWindow window{ sf::VideoMode(800, 600), "complex_using", sf::Style::Close };

  while (true)
  {
    sf::Event event;
    while (window.pollEvent(event) == true)
    {
      if (event.type == sf::Event::EventType::Closed)
      {
        return 0;
      }
    }
    window.clear();
    //window.draw(something);
    window.display();
  }

  return 0;
}