#ifndef TRACKED_OBJ_H
#define TRACKED_OBJ_H

template <typename T>
class CounterObj {
   public:
    static size_t objects_made;
    static size_t objects_active;

    CounterObj() {
        ++objects_made;
        ++objects_active;
    }

    CounterObj(const Counter &) {
        ++objects_made;
        ++objects_active;
    }

   protected:
    ~CounterObj() { --objects_active; }
};

template <typename T>
size_t CounterObj<T>::objects_made{0};
template <typename T>
size_t CounterObj<T>::objects_active{0};

#endif
