#include <iostream>
#include <vector>
#include <limits>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

const long long INF = numeric_limits<long long>::max();

struct Edge {
    int to;
    long long weight;
};

vector<long long> dijkstra_parallel(int start, const vector<vector<Edge>>& graph) {
    int n = graph.size();
    vector<long long> dist(n, INF);
    vector<bool> visited(n, false);
    dist[start] = 0;
    bool work_remaining = true;

    while (work_remaining) {
        work_remaining = false;

#pragma omp parallel for
        for (int u = 0; u < n; ++u) {
            if (!visited[u] && dist[u] < INF) {
                visited[u] = true;

                for (const auto& edge : graph[u]) {
                    int v = edge.to;
                    long long weight = edge.weight;

                    if (dist[u] + weight < dist[v]) {
#pragma omp critical
                        {
                            if (dist[u] + weight < dist[v]) {
                                dist[v] = dist[u] + weight;
                                work_remaining = true;
                            }
                        }
                    }
                }
            }
        }
    }
    return dist;
}

int main() {
    srand(time(0));

    int nodes = 100000;
    int edges = 1000000;
    int maxWeight = 10000000;
    int startNode = 0;

    auto start_gen = high_resolution_clock::now();

    ofstream out("graph.txt");
    if (!out.is_open()) {
        cerr << "Error: Could not create graph.txt" << endl;
        return 1;
    }

    out << nodes << " " << edges << "\n";

    for (int i = 0; i < edges; ++i) {
        int u = rand() % nodes;
        int v = rand() % nodes;
        while (v == u) v = rand() % nodes;
        int weight = 1 + rand() % maxWeight;
        out << u << " " << v << " " << weight << "\n";
    }
    out.close();

    cout << "Graph written to graph.txt" << endl;

    auto end_gen = high_resolution_clock::now();

    auto start_read = high_resolution_clock::now();

    ifstream infile("graph.txt");
    if (!infile.is_open()) {
        cerr << "Error: Could not open graph.txt" << endl;
        return 1;
    }

    infile >> nodes >> edges;
    vector<vector<Edge>> graph(nodes);

    for (int i = 0; i < edges; ++i) {
        int u, v;
        long long w;
        infile >> u >> v >> w;
        graph[u].push_back({ v, w });
    }
    infile.close();

    cout << "Graph loaded. Running Dijkstra from node " << startNode << "..." << endl;

    auto end_read = high_resolution_clock::now();

    auto start_algo = high_resolution_clock::now();
    vector<long long> distances = dijkstra_parallel(startNode, graph);
    auto end_algo = high_resolution_clock::now();

    ofstream outfile("distances.txt");
    if (!outfile.is_open()) {
        cerr << "Error: Could not open distances.txt" << endl;
        return 1;
    }

    for (int i = 0; i < nodes; ++i) {
        if (distances[i] == INF)
            outfile << "Node " << i << ": -1\n";
        else
            outfile << "Node " << i << ": " << distances[i] << "\n";
    }
    outfile.close();

    cout << "Distances written to distances.txt" << endl;

    cout << fixed;
    cout << "Graph generation:    " << duration<double>(end_gen - start_gen).count() << " sec\n";
    cout << "Graph reading:      " << duration<double>(end_read - start_read).count() << " sec\n";
    cout << "Parallel Dijkstra: " << duration<double>(end_algo - start_algo).count() << " sec\n";

    return 0;
}
