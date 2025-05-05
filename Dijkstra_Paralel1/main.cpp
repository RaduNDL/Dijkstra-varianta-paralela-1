#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <unordered_set>
#include <climits>
#include <sstream>
#include <omp.h>
using namespace std;
const int INF = INT_MAX;
const int MAX_WEIGHT = 100;
const int NUM_THREADS = 8;  
inline uint64_t hash_edge(int u, int v, int V) {
	return (uint64_t)min(u, v) * V + max(u, v);
}
void generate_graph(int V, int E, const string& filename) {
	ofstream fout(filename);
	fout << V << " " << E << "\n";
	mt19937 rng(123);
	uniform_int_distribution<int> dist_node(0, V - 1);
	uniform_int_distribution<int> dist_weight(1, MAX_WEIGHT);
	unordered_set<uint64_t> edges;
	edges.reserve(E);
	vector<string> edge_list;
	edge_list.reserve(E);
	int edge_count = 0;
	for (int i = 1; i < V; ++i) {
		int u = i - 1;
		int v = i;
		int w = dist_weight(rng);
		uint64_t h = hash_edge(u, v, V);
		if (edges.insert(h).second) {
			ostringstream oss;
			oss << u << " " << v << " " << w << "\n";
			edge_list.push_back(oss.str());
			edge_count++;
		}
	}
	while (edge_count < E) {
		int u = dist_node(rng);
		int v = dist_node(rng);
		if (u == v) continue;
		uint64_t h = hash_edge(u, v, V);
		if (edges.insert(h).second) {
			int w = dist_weight(rng);
			ostringstream oss;
			oss << u << " " << v << " " << w << "\n";
			edge_list.push_back(oss.str());
			edge_count++;
		}
	}
	for (const auto& edge : edge_list) {
		fout << edge;
	}
	fout.close();
	cout << "Graph successfully generated in " << filename << "\n";
}
void dijkstra_openmp(int V, const vector<vector<pair<int, int>>>& adj, int src, const string& filename) {
	vector<int> dist(V, INF);
	dist[src] = 0;
	int max_dist = MAX_WEIGHT * V;
	vector<vector<int>> buckets(max_dist + 1);
	buckets[0].push_back(src);
	int curr = 0;
	while (curr <= max_dist) {
		if (buckets[curr].empty()) {
			++curr;
			continue;
		}
		vector<int> this_bucket;
		swap(this_bucket, buckets[curr]);
#pragma omp parallel for schedule(dynamic) shared(dist, buckets) 
		for (int i = 0; i < this_bucket.size(); ++i) {
			int u = this_bucket[i];
			for (const auto& [v, w] : adj[u]) {
				int newDist = dist[u] + w;
#pragma omp critical
				{
					if (newDist < dist[v]) {
						dist[v] = newDist;
						if (newDist <= max_dist) {						
#pragma omp critical
							buckets[newDist].push_back(v);
						}
					}
				}
			}
		}
	}
	ofstream fout(filename);
	for (int i = 0; i < V; ++i) {
		fout << i << " " << (dist[i] == INF ? -1 : dist[i]) << "\n";
	}
	fout.close();
}
int main() {
	ios::sync_with_stdio(false);
	cin.tie(nullptr);
	long V = 10000;
	long E = 100000;
	string input_file = "input.txt";
	string output_file = "distances.txt";
	auto start_gen = chrono::high_resolution_clock::now();
	generate_graph(V, E, input_file);  
	auto end_gen = chrono::high_resolution_clock::now();
	auto start_read = chrono::high_resolution_clock::now();
	ifstream fin(input_file);
	int file_V, file_E;
	fin >> file_V >> file_E;
	vector<vector<pair<int, int>>> adj(file_V);
	for (int i = 0; i < file_E; ++i) {
		int u, v, w;
		fin >> u >> v >> w;
		adj[u].emplace_back(v, w);
	}
	fin.close();
	auto end_read = chrono::high_resolution_clock::now();
	auto start_algo = chrono::high_resolution_clock::now();
	dijkstra_openmp(file_V, adj, 0, output_file);  
	auto end_algo = chrono::high_resolution_clock::now();
	cout << "Graph Generation: " << chrono::duration<double>(end_gen - start_gen).count() << " sec\n";
	cout << "Graph Reading:    " << chrono::duration<double>(end_read - start_read).count() << " sec\n";
	cout << "Dijkstra:         " << chrono::duration<double>(end_algo - start_algo).count() << " sec\n";
	cout << "Distances written to: " << output_file << "\n";
	return 0;
}