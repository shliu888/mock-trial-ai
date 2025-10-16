import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
llm = init_chat_model("google_genai:gemini-2.0-flash")

from typing_extensions import TypedDict

from langgraph.graph import StateGraph,END
from langchain_core.messages import SystemMessage, HumanMessage


class State(TypedDict):
    usr_prompt: str
    scenario: str
    lawyer1_argument : str
    prosecutor1_argument: str
    lawyer2_argument: str
    prosecutor2_argument: str
    verdict: str


graph_builder = StateGraph(State)
def lawyer1(state: State):

    sys_prompt=SystemMessage(content="You are an experienced lawyer who is trying to prove that the defendant is innocent. You are ONLY to use existing laws and the existing evidence, and you MUST try your best to defend the defendant, no matter how convincing the prosecutor is. You are to debate with the prosecutor based on their arguments. Cross-examination is NOT allowed.")
    msg=[sys_prompt,HumanMessage(content=f"Scenario: {state['scenario']}")]
    argument=llm.invoke(msg).content
    return {"lawyer1_argument": argument}

def lawyer2(state: State):
    sys_prompt=SystemMessage(content="You are an experienced lawyer who is trying to prove that the defendant is innocent. You are ONLY to use existing laws and the existing evidence, and you MUST try your best to defend the defendant, no matter how convincing the prosecutor is. You are to debate with the prosecutor based on their arguments. Cross-examination is NOT allowed. You are now in Round 2 and must provide a rebuttal to the prosecutor's argument")
    msg=[sys_prompt,HumanMessage(content=f"Scenario: {state['scenario']}, Prosecutor's argument: {state['prosecutor1_argument']}")]
    argument=llm.invoke(msg).content
    return {"lawyer2_argument": argument}
def prosecutor1(state: State):
    sys_prompt=SystemMessage(content="You are an experienced prosecutor who is trying to prove that the defendant is guilty. You are ONLY to use existing laws and the existing evidence, and you MUST try your best to prove that the defendant is guilty, no matter how convincing the lawyer is. You are to debate with the lawyer based on their arguments. Cross-examination is NOT allowed.")
    msg=[sys_prompt,HumanMessage(content=f"Lawyer argument: {state['lawyer1_argument']}, Scenario: {state['scenario']}")]
    argument=llm.invoke(msg).content
    return {"prosecutor1_argument": argument}
def prosecutor2(state: State):
    sys_prompt=SystemMessage(content="You are an experienced prosecutor who is trying to prove that the defendant is guilty. You are ONLY to use existing laws and the existing evidence, and you MUST try your best to prove that the defendant is guilty, no matter how convincing the lawyer is. You are to debate with the lawyer based on their arguments. Cross-examination is NOT allowed.")
    msg=[sys_prompt,HumanMessage(content=f"Lawyer argument: {state['lawyer2_argument']}, Scenario: {state['scenario']}")]
    argument=llm.invoke(msg).content
    return {"prosecutor2_argument": argument}
def scenario_generator(state: State):
    sys_prompt=SystemMessage("You are an expert scenario generator. You are to take the prompt that the user gives you, and generates it into a cohesive mock trial setup. In the scenario that you generate, you are to not only generate the crime that the defendant is accused of and the storyline, but generate evidence, some of which acts in support of the defendant, and some of which acts against the defendant. You shall not give the lawyer and prosecutor names, just refer to them as 'the lawyer' and 'the prosecutor")
    msg=[sys_prompt, HumanMessage(content=f"Scenario: {state['usr_prompt']}")]
    gen_scenario=llm.invoke(msg).content
    return {"scenario": gen_scenario}
def judge(state: State):
    sys_prompt=SystemMessage("You are a judge in a court, deciding if the defendant is guilty or not (presenting your verdict). Your job is to judge NEUTRALLY and using ONLY arguments presented by the lawyer and the prosecutor and the evidence in the scenario. You are to list out your chain of thought, and make sure to have THOROUGH REASONING. ")
    msg=[sys_prompt,HumanMessage(content=f"Scenario: {state['scenario']}, Lawyer's argument #1: {state['lawyer1_argument']}, Prosecutor's argument #1: {state['prosecutor1_argument']}, Lawyer's argument #2: {state['lawyer2_argument']}, Prosecutor's argument #2: {state['prosecutor2_argument']}")]
    decision=llm.invoke(msg).content
    return {"verdict": decision}
while True:
    user_input = input("Enter your prompt for the mock trial ('exit' to stop): ")
    if user_input.lower() == 'exit':
        print("Exiting.")
        break
    graph_builder = StateGraph(State)
    
    graph_builder.add_node("scenario", scenario_generator)
    graph_builder.add_node("lawyer1", lawyer1)
    graph_builder.add_node("prosecutor1", prosecutor1)
    graph_builder.add_node("lawyer2",lawyer2)
    graph_builder.add_node("prosecutor2",prosecutor2)
    graph_builder.add_node("judge",judge)

    graph_builder.set_entry_point("scenario") 
    graph_builder.add_edge("scenario", "lawyer1")
    graph_builder.add_edge("lawyer1", "prosecutor1")
    graph_builder.add_edge("prosecutor1","lawyer2" )
    graph_builder.add_edge("lawyer2","prosecutor2")
    graph_builder.add_edge("prosecutor2","judge")
    graph_builder.add_edge("judge",END)
    
    app = graph_builder.compile()
    initial_state = {"usr_prompt": user_input} 
    result = app.invoke(initial_state) 
    
    print("\nScenario:\n", result["scenario"])
    print("\n--- LAWYER ARGUMENT ---\n", result["lawyer1_argument"])
    print("\n--- PROSECUTOR ARGUMENT ---\n", result["prosecutor1_argument"])
    print("\n--- LAWYER ARGUMENT ---\n", result["lawyer2_argument"])
    print("\n--- PROSECUTOR ARGUMENT ---\n", result["prosecutor2_argument"])
    print("\n--- VERDICT ---\n", result["verdict"])
    print("\n" + "="*50)
