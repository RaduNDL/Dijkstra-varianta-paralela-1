#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <climits>
#include <chrono>
#include <thread>
#include <mutex>
using namespace std;

const int INF = INT_MAX;
mutex pq_mutex;

void parallel_dijkstra(int V, vector<vector<pair<int, int>>>& adj, int src, const string& filename, int num_threads) {
    vector<int> dist(V, INF);
    dist[src] = 0;

    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
    pq.push({ 0, src });

    auto worker = [&](int tid) {
        while (true) {
            pq_mutex.lock();
            if (pq.empty()) {
                pq_mutex.unlock();
                break;
            }
            auto [d, u] = pq.top(); pq.pop();
            pq_mutex.unlock();

            if (d > dist[u]) continue;

            for (auto [v, w] : adj[u]) {
                int new_dist = dist[u] + w;
                if (new_dist < dist[v]) {
                    dist[v] = new_dist;
                    pq_mutex.lock();
                    pq.push({ new_dist, v });
                    pq_mutex.unlock();
                }
            }
        }
        };

    vector<thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    ofstream fout(filename);
    for (int i = 0; i < V; ++i)
        fout << i << " " << (dist[i] == INF ? -1 : dist[i]) << "\n";
    fout.close();
}

int main() {
    ifstream fin("graph.txt");
    if (!fin) {
        cerr << "Fisierul graph.txt nu exista!\n";
        return 1;
    }

    int V, E;
    fin >> V >> E;
    vector<vector<pair<int, int>>> adj(V);

    for (int i = 0; i < E; ++i) {
        int u, v, w;
        fin >> u >> v >> w;
        adj[u].push_back({ v, w });
    }

    fin.close();

    auto start = chrono::high_resolution_clock::now();
    parallel_dijkstra(V, adj, 0, "distances_parallel.txt", thread::hardware_concurrency());
    auto end = chrono::high_resolution_clock::now();

    cout << "Dijkstra paralel a terminat in " << chrono::duration<double>(end - start).count() << " secunde\n";
    return 0;
}