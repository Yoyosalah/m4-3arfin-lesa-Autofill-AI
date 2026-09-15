from langgraph.graph import StateGraph, START, END
from .schemas import AppState
from .nodes import ocr_parsing_node, stt_node, extraction_node, validation_node, should_loop, check_fatal_error

builder = StateGraph(AppState)

# Nodes
builder.add_node("ocr_parsing_node", ocr_parsing_node)
builder.add_node("stt_node", stt_node)
builder.add_node("extraction_node", extraction_node)
builder.add_node("validation_node", validation_node)

# Flow
builder.add_edge(START, "ocr_parsing_node")

builder.add_conditional_edges(
    "ocr_parsing_node",
    check_fatal_error,
    {"continue": "stt_node", "finish": END}
)

builder.add_conditional_edges(
    "stt_node",
    check_fatal_error,
    {"continue": "extraction_node", "finish": END}
)

builder.add_conditional_edges(
    "extraction_node",
    check_fatal_error,
    {"continue": "validation_node", "finish": END}
)

builder.add_conditional_edges(
    "validation_node", 
    should_loop, 
    {"rewrite": "extraction_node", "finish": END}
)

app_graph = builder.compile()