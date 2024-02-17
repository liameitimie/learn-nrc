#include <index_heap.h>
#include <luisa/core/mathematics.h>
#include <assert.h>

using namespace luisa;

IndexHeap::IndexHeap(int num_index) {
    index_in_heap.resize(num_index);
    memset(index_in_heap.data(), 0xff, num_index * sizeof(int));

    _keys.reserve(num_index);
    heap.reserve(num_index);
}

void IndexHeap::push_up(int i) {
    int idx = heap[i];
    double key = _keys[i];
    int fa = (i - 1) >> 1;
    while (i > 0 && key < _keys[fa]) {
        heap[i] = heap[fa];
        _keys[i] = _keys[fa];
        index_in_heap[heap[i]] = i;
        i = fa, fa = (i - 1) >> 1;
    }
    heap[i] = idx;
    _keys[i] = key;
    index_in_heap[heap[i]] = i;
}

void IndexHeap::push_down(int i) {
    int idx = heap[i];
    double key = _keys[i];
    int ls = (i << 1) + 1;
    int rs = ls + 1;
    int heap_size = heap.size();
    while (ls < heap_size) {
        int t = ls;
        if (rs < heap_size && _keys[rs] < _keys[ls])
            t = rs;
        if (_keys[t] < key) {
            heap[i] = heap[t];
            _keys[i] = _keys[t];
            index_in_heap[heap[i]] = i;
            i = t, ls = (i << 1) + 1, rs = ls + 1;
        }
        else break;
    }
    heap[i] = idx;
    _keys[i] = key;
    index_in_heap[heap[i]] = i;
}

void IndexHeap::update(int index, double key) {
    int &i = index_in_heap[index];
    if (i == -1) {
        _keys.push_back(key);
        heap.push_back(index);
        i = heap.size() - 1;

        push_up(i);
    }
    else {
        _keys[i] = key;
        adjust(i);
    }
}

void IndexHeap::remove(int index) {
    int i = index_in_heap[index];

    if (i == -1) return;
    if (i == heap.size() - 1) {
        index_in_heap[index] = -1;
        heap.pop_back();
        _keys.pop_back();
        return;
    }

    double key = _keys[i];

    heap[i] = heap.back();
    heap.pop_back();
    _keys[i] = _keys.back();
    _keys.pop_back();
    index_in_heap[heap[i]] = i;
    index_in_heap[index] = -1;

    if (key < _keys[i]) push_down(i);
    else push_up(i);
}

void IndexHeap::adjust(int i) {
    if (i > 0 && _keys[i] < _keys[(i - 1) >> 1]) {
        push_up(i);
    } else {
        push_down(i);
    }
}

bool IndexHeap::is_present(int index) {
    return index_in_heap[index] != -1;
}

// void IndexHeap::update_unadjust(int index, double key) {
//     _keys[index] = key;
// }

// void IndexHeap::build_heap() {
//     heap.resize(_keys.size());
//     for (int i = 0; i < _keys.size(); i++) {
//         heap[i] = i;
//         index_in_heap[i] = i;
//     }
//     for (int i = next_pow2(_keys.size() + 1) / 2 - 2; i >= 0; i--) {
//         push_down(i);
//     }
// }

IndexHeap::KeyWithIndex IndexHeap::top() {
    assert(heap.size() > 0);
    return { _keys[0], heap[0] };
}

IndexHeap::KeyWithIndex IndexHeap::pop() {
    assert(heap.size() > 0);

    IndexHeap::KeyWithIndex res = { _keys[0], heap[0] };

    if (heap.size() == 1) {
        index_in_heap[heap[0]] = -1;
        heap.pop_back();
        _keys.pop_back();
    }
    else {
        index_in_heap[heap[0]] = -1;
        heap[0] = heap.back();
        heap.pop_back();
        _keys[0] = _keys.back();
        _keys.pop_back();
        index_in_heap[heap[0]] = 0;

        push_down(0);
    }
    return res;
}