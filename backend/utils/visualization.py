import networkx as nx
from pyvis.network import Network
from typing import List, Union
from llama_index.core.schema import IndexNode, TextNode
import logging


# Function to truncate text and add line breaks
def truncate_text(text, max_length=180):
    return "\n".join(
        [text[i : i + max_length] for i in range(0, len(text), max_length)]
    )


def find_src_node_id(nodes, node_ref_doc_id: str):
    try:
        return nodes[[n.ref_doc_id == node_ref_doc_id for n in nodes].index(True)].id_
    except Exception as e:
        return None


def visualize_nodes_with_attributes(
    nodes: List[Union[IndexNode, TextNode]], graph_name_prefix: str = ""
):
    G = nx.Graph()

    # Add nodes to the graph with attributes
    for i, node in enumerate(nodes):
        n_id = f"i_{i} || ref_doc_id_part_{node.ref_doc_id.split('_')[-1]}"
        G.add_node(
            node.id_,
            # graph_id=node.id,
            id_=node.id_,
            node_id=node.node_id,
            text=node.text,
            end_chr_idx=node.end_char_idx,
            start_char_idx=node.start_char_idx,
            label=n_id,
            mimetype=node.mimetype,
            ref_doc_id=node.ref_doc_id,
        )

    # Add edges based on relationships
    for node in nodes:
        for r_type, r_info in node.relationships.items():
            if r_type.name == "SOURCE":
                # almost all nodes have SOURCE relationship
                src_node_id = find_src_node_id(nodes, r_info.node_id)
                if src_node_id:  # can found the src node in the node list
                    if (
                        src_node_id != node.id_
                    ):  # only add relation when it's not referring to itself
                        G.add_edge(src_node_id, node.id_, label=r_type.name)
                else:  # can't find the src node in the node list, add new node in graph
                    G.add_edge(r_info.node_id, node.id_, label=r_type.name)
            elif r_type.name in ["PARENT", "PREVIOUS"]:
                # add relationship from parent -> child, or previous -> next
                G.add_edge(r_info.node_id, node.id_, label=r_type.name)
            else:
                G.add_edge(node.id_, r_info.node_id, label=r_type.name)

    # Convert the NetworkX graph to a PyVis graph
    net = Network(
        notebook=False,
        cdn_resources="in_line",
        width="100%",
        height="1500px",
        bgcolor="#ffffff",
        font_color="black",
    )
    net.from_nx(G)

    # Customize the appearance of the nodes and edges
    net.set_options(
        """
        var options = {
          "nodes": {
            "borderWidth": 2,
            "shape": "dot",
            "size": 20,
            "color": {
              "border": "#2B7CE9",
              "background": "#97C2FC"
            },
            "font": {
              "size": 14,
              "color": "#343434"
            },
            "shadow": {
              "enabled": true
            }
          },
          "edges": {
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 1
              }
            },
            "color": {
              "color": "#848484",
              "highlight": "#848484",
              "inherit": false,
              "opacity": 0.6
            },
            "smooth": {
              "enabled": true,
              "type": "dynamic"
            }
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          },
          "physics": {
            "enabled": true,
            "stabilization": {
              "enabled": true,
              "fit": true,
              "iterations": 1000,
              "onlyDynamicEdges": false,
              "updateInterval": 50
            }
          }
        }
        """
    )

    # Customize node hover information
    for node in net.nodes:
        node["title"] = (
            f"Graph ID: {node['id']}\n"
            f"ID_: {node.get('id_', 'N/A')}\n"
            f"Node ID: {node.get('node_id', 'N/A')}\n"
            f"Text: {truncate_text(node.get('text', 'N/A'))}\n"
            f"End Char Index: {node.get('end_chr_idx', 'N/A')}\n"
            f"Start Char Index: {node.get('start_char_idx', 'N/A')}\n"
            f"Label: {node.get('label', 'N/A')}\n"
            f"MIME Type: {node.get('mimetype', 'N/A')}\n"
            f"Ref Doc ID: {node.get('ref_doc_id', 'N/A')}"
        )

    # Visualize the graph
    net.save_graph(f"{graph_name_prefix}_graph.html")
