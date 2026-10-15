// Weak baseline for Frontier-CS algorithmic problem 0 (Polyomino Packing).
//
// Idea: treat every polyomino as its axis-aligned bounding box, turned so the box is flat
// (width >= height; no reflections), sort boxes by height (desc), pack them with First-Fit
// Decreasing Height shelves for a range of candidate strip widths, keep the smallest W*H.
// Deterministic: no randomness, no wall-clock dependence. Single C++17 file.
#include <bits/stdc++.h>
using namespace std;

struct Box {
    int id;
    int w, h;     // bounding box after the chosen rotation
    int R;        // number of 90-degree clockwise rotations (0 or 1)
    int offx, offy;  // -min(x), -min(y) of the rotated cells
};

static inline pair<int, int> rot90cw(int x, int y, int r) {
    switch (r & 3) {
        case 0: return {x, y};
        case 1: return {y, -x};
        case 2: return {-x, -y};
        default: return {-y, x};
    }
}

struct Packing {
    long long area = LLONG_MAX;
    int W = 0, H = 0;
    vector<pair<int, int>> pos;  // per sorted box: lower-left corner of its bounding box
};

// First-Fit Decreasing Height shelf packing into a strip of width `width`.
static Packing shelfPack(const vector<Box>& boxes, int width) {
    Packing p;
    p.pos.resize(boxes.size());
    vector<int> shelfY, shelfH, shelfUsed;
    int totalH = 0, usedW = 0;
    for (size_t i = 0; i < boxes.size(); ++i) {
        const Box& b = boxes[i];
        int s = -1;
        for (size_t j = 0; j < shelfY.size(); ++j) {
            if (shelfUsed[j] + b.w <= width && b.h <= shelfH[j]) { s = (int)j; break; }
        }
        if (s < 0) {
            shelfY.push_back(totalH);
            shelfH.push_back(b.h);
            shelfUsed.push_back(0);
            totalH += b.h;
            s = (int)shelfY.size() - 1;
        }
        p.pos[i] = {shelfUsed[s], shelfY[s]};
        shelfUsed[s] += b.w;
        usedW = max(usedW, shelfUsed[s]);
    }
    p.W = usedW;
    p.H = totalH;
    p.area = (long long)usedW * totalH;
    return p;
}

int main() {
    int n;
    if (scanf("%d", &n) != 1) return 0;
    vector<Box> boxes(n);
    long long boxArea = 0;
    int maxW = 1;
    for (int i = 0; i < n; ++i) {
        int k;
        scanf("%d", &k);
        vector<pair<int, int>> cells(k);
        for (auto& c : cells) scanf("%d %d", &c.first, &c.second);
        Box best{};
        best.w = INT_MAX;
        for (int r = 0; r < 2; ++r) {
            int mnx = INT_MAX, mny = INT_MAX, mxx = INT_MIN, mxy = INT_MIN;
            for (auto& c : cells) {
                auto [x, y] = rot90cw(c.first, c.second, r);
                mnx = min(mnx, x); mny = min(mny, y);
                mxx = max(mxx, x); mxy = max(mxy, y);
            }
            int w = mxx - mnx + 1, h = mxy - mny + 1;
            // prefer the flat orientation (smaller height); tie -> r = 0
            if (r == 0 || h < best.h) best = Box{i, w, h, r, -mnx, -mny};
        }
        boxes[i] = best;
        boxArea += (long long)best.w * best.h;
        maxW = max(maxW, best.w);
    }

    vector<Box> sorted = boxes;
    stable_sort(sorted.begin(), sorted.end(), [](const Box& a, const Box& b) {
        if (a.h != b.h) return a.h > b.h;
        return a.w > b.w;
    });

    // Candidate strip widths around sqrt(total box area).
    int side = max(maxW, (int)ceil(sqrt((double)boxArea)));
    vector<int> widths;
    for (int pct = 60; pct <= 160; pct += 4) {
        int w = max(maxW, (int)llround(side * pct / 100.0));
        if (widths.empty() || widths.back() != w) widths.push_back(w);
    }

    Packing best;
    for (int w : widths) {
        Packing p = shelfPack(sorted, w);
        if (p.area < best.area) best = std::move(p);
    }

    vector<array<int, 4>> out(n);
    for (size_t i = 0; i < sorted.size(); ++i) {
        const Box& b = sorted[i];
        out[b.id] = {best.pos[i].first + b.offx, best.pos[i].second + b.offy, b.R, 0};
    }
    printf("%d %d\n", best.W, best.H);
    for (int i = 0; i < n; ++i) printf("%d %d %d %d\n", out[i][0], out[i][1], out[i][2], out[i][3]);
    return 0;
}
