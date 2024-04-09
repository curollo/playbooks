#!/usr/bin/env python3
import re
import argparse
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import argparse

def get_nodes(tokens: list[str], prune=0) -> nx.Graph:
    nodes = list(set(tokens))
    edges = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
    edge_array, counts = np.unique(edges, return_counts=1, axis=0)
    edge_keys = [tuple(x) for x in edge_array]
    edge_counter = dict(zip(edge_keys, counts))
    #print(edge_counter)

    G = nx.Graph()
    G.add_nodes_from(nodes)

    for edge in edge_counter:
        G.add_edge(edge[0], edge[1], weight=edge_counter[edge]*10)

    if prune:
        # remove if node degree < 3
        nodes_to_remove = [node for node in G.nodes if G.degree[node] < 3]
        G.remove_nodes_from(nodes_to_remove)

    # assign text to nodes
    for node in G.nodes: G.nodes[node]["text"] = node

    # node size
    scale = 10
    d = dict(G.degree)
    d = {k: v * scale for k, v in d.items()}
    nx.set_node_attributes(G, d, "size")
    #print(G.edges.data())

    return G

def get_tokens(tokens):
    # conv string tokens into integer indices
    _, idx = np.unique(tokens, return_inverse=1)

    # store the frequency distribution in a dictionary
    freq = {tokens[i]: c for i, c in enumerate(np.bincount(idx))}

    # sort dict by value (frequency)
    return dict(sorted(freq.items(), key=lambda item: item[1], reverse=1))


def split(text: str, n: int, words=1) -> list[str]:
    # drop non-alphanumeric, convert to lowercase
    text = re.sub(r'\W+', ' ', text).lower()

    # percentage of used text
    if 0 < n <= 100: txt_len = int(len(text) * np.divide(n, 100))
    else: raise ValueError("Percentage should be between 0 and 100.")

    # use letters or entire words
    if words: return text[:txt_len].split()
    else: return list(text[:txt_len])

#text = 'co tu sie dzieje'

def main():
    parser = argparse.ArgumentParser(description='process some inputs.')
    parser.add_argument('-f', '--file', type=str, help='load text from a file')
    parser.add_argument('-i', '--input', action='store_true', help='get text input prompt')
    parser.add_argument('-w', '--words', required=0, help='words or single letters [Words by default]')
    parser.add_argument('-n', '--length', required=0, type=int, help='percentage or used text')
    parser.add_argument('-t', '--tokens', required=0, type=int, help='plot by given number of tokens')
    args = parser.parse_args()

    if args.file:
        with open(args.file, 'r') as f:
            text = f.read()
    elif args.input:
        text = input("Enter your text: ")
    else:
        print("No valid arguments provided.")
        exit(1)

    print("Text:", text)

    if args.length is None: args.length = 100

    tokens = split(text, n=args.length, words=args.words)
    token_counts = get_tokens(tokens)

    if args.tokens is not None:
        keys = list(token_counts.keys())
        slice_keys = keys[:args.tokens]
        token_counts = {k: token_counts[k] for k in slice_keys}

    colnames = ['token', 'count']
    df = pd.DataFrame(token_counts.items(), columns=colnames)
    df = df.sort_values('count', ascending=0)
    df = df.reset_index(drop=1)

    if args.tokens: df = df[:args.tokens]

    # convert 'token' column into categorical dtype
    categories = list(df['token'])
    df['token'] = pd.Categorical(df['token'], categories=categories)

    # set 'token' as the index and reset the original index
    df.set_index('token', drop=True, inplace=True)
    df.index = df.index.astype(str) # cast the index to str

    df['count'].plot(kind='bar', rot=0, width=0.75)
    plt.xlabel("Token")
    plt.ylabel("Count")
    plt.title("Histogram of Token Counts")
    plt.savefig("histogram.png", format="PNG")
    plt.figure(0)
    #plt.show()

    df["count [log]"] = df["count"].apply(np.log)
    df = df.reset_index(drop=1)
    df["index"] = np.log(df.index+1)
    plt.scatter(df['index'], df["count [log]"])
    plt.xlabel('index')
    plt.ylabel('count')
    plt.title('Log-Histogram')
    plt.savefig("log_histogram.png", format="PNG")
    plt.figure(2)
    #plt.show()

    if args.tokens is not None:
        G = get_nodes(list(token_counts.keys()), prune=0)
    else:
        G = get_nodes(tokens, prune=0)

    nx.draw(G, with_labels=1, node_color='lightblue', edge_color='gray')
    plt.savefig("graph.png", format="PNG")
    plt.figure(3)
    #plt.show()

if __name__ == "__main__":
    main()
