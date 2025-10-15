import os
from langchain.chat_models import init_chat_model

os.environ["GOOGLE_API_KEY"] = "AIzaSyDw1ju70N8qm0LGWFHWwVWGoiKpq-626dY"

llm = init_chat_model("google_genai:gemini-2.0-flash")
from typing_extensions import TypedDict

from langgraph.graph import StateGraph,END
from langchain_core.messages import SystemMessage, HumanMessage


class State(TypedDict):
    usr_prompt: str
    scenario: str
    lawyer_argument : str
    prosecutor_argument: str


graph_builder = StateGraph(State)
def lawyer(state: State):
    sys_prompt=SystemMessage(content="You are an experienced lawyer who is trying to prove that the defendant is innocent. You are ONLY to use existing laws and the existing evidence, and you MUST try your best to defend the defendant, no matter how convincing the prosecutor is. You are to debate with the prosecutor based on their arguments. Cross-examination is NOT allowed.")
    msg=[sys_prompt,HumanMessage(content=f"Scenario: {state['scenario']}")]
    argument=llm.invoke(msg).content
    return {"lawyer_argument": argument}


def prosecutor(state: State):
    sys_prompt=SystemMessage(content="You are an experienced prosecutor who is trying to prove that the defendant is guilty. You are ONLY to use existing laws and the existing evidence, and you MUST try your best to prove that the defendant is guilty, no matter how convincing the lawyer is. You are to debate with the lawyer based on their arguments. Cross-examination is NOT allowed.")
    msg=[sys_prompt,HumanMessage(content=f"Lawyer argument: {state['lawyer_argument']}, Scenario: {state['scenario']}")]
    argument=llm.invoke(msg).content
    return {"prosecutor_argument": argument}

def scenario_generator(state: State):
    sys_prompt=SystemMessage("You are an expert scenario generator. You are to take the prompt that the user gives you, and generates it into a cohesive mock trial setup. In the scenario that you generate, you are to not only generate the crime that the defendant is accused of and the storyline, but generate evidence, some of which acts in support of the defendant, and some of which acts against the defendant.")
    msg=[sys_prompt, HumanMessage(content=f"Scenario: {state['usr_prompt']}")]
    gen_scenario=llm.invoke(msg).content
    return {"scenario": gen_scenario}

while True:
    user_input = input("Enter your prompt for the mock trial ('exit' to stop): ")
    if user_input.lower() == 'exit':
        print("Exiting.")
        break
    graph_builder = StateGraph(State)
    
    graph_builder.add_node("scenario", scenario_generator)
    graph_builder.add_node("lawyer", lawyer)
    graph_builder.add_node("prosecutor", prosecutor)

    graph_builder.set_entry_point("scenario") 
    graph_builder.add_edge("scenario", "lawyer")
    graph_builder.add_edge("lawyer", "prosecutor")
    graph_builder.add_edge("prosecutor", END)
    
    app = graph_builder.compile()
    initial_state = {"usr_prompt": user_input} 
    result = app.invoke(initial_state) 
    
    print("\nScenario:\n", result["scenario"])
    print("\n--- LAWYER ARGUMENT ---\n", result["lawyer_argument"])
    print("\n--- PROSECUTOR ARGUMENT ---\n", result["prosecutor_argument"])
    print("\n" + "="*50)
