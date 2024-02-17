#pragma once

#include <luisa/core/stl/vector.h>

// 对下标建堆，通过更新key调整堆

class IndexHeap {
    luisa::vector<double> _keys;
    luisa::vector<int> heap; // 存储下标
    luisa::vector<int> index_in_heap; // 存储下标在堆中的位置，当下标不在堆中时为-1

    void push_up(int i);
    void push_down(int i);
    void adjust(int i); // 调整堆

public:
    IndexHeap(int num_index);

    void update(int index, double key);
    void remove(int index);

    double key(int index) { return _keys[index_in_heap[index]]; }

    bool is_present(int index);

    struct KeyWithIndex {
        double key;
        int index;
    };

    KeyWithIndex top();
    KeyWithIndex pop();

    int size() { return heap.size(); }
};