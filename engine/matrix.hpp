#pragma once

#include <memory>
#include <stdexcept>

namespace cnn
{
  template <typename T>
  class Matrix
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    Matrix(const size_t count);

    size_t GetCount() const;

    T GetCell(const size_t index) const;
    void SetCell(const size_t index, const T value);

  private:

    const size_t Count;
    std::unique_ptr<T[]> Cells;

  };

  template <typename T>
  Matrix<T>::Matrix(const size_t count)
    :
    Count{ count },
    Cells{ std::make_unique<T[]>(Count) }
  {
    if (Count == 0)
    {
      throw std::invalid_argument("cnn::Matrix::Matrix(), Count == 0.");
    }
    memset(Cells.get(), 0, Count * sizeof(T));
  }

  template <typename T>
  size_t Matrix<T>::GetCount() const
  {
    return Count;
  }

  template <typename T>
  T Matrix<T>::GetCell(const size_t index) const
  {
    if (index >= Count)
    {
      throw std::range_error("cnn::Matrix::GetCell(), index >= Count.");
    }
    return Cells[index];
  }

  template <typename T>
  void Matrix<T>::SetCell(const size_t index, const T value)
  {
    if (index >= Count)
    {
      throw std::range_error("cnn::Matrix::SetCell(), index >= Count.");
    }
    Cells[index] = value;
  }
}
