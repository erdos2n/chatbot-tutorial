import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from frontend.model_loader import load_clf_bow

clf, bow = load_clf_bow()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

app.layout = html.Div([
    dcc.Input(id="input-box", placeholder="insert chat here", type="text"),
    html.Br(),
    dbc.Button("Submit", id="button"),
    html.Div(id="output-box"),
    html.Div(id="answer-box")
])

@app.callback(Output("output-box", "children"),
              [Input("input-box", "value")])
def render_output_box(text):
    if text == "" or text is None:
        text = "Start Conversation"
    return html.H2(text)


@app.callback(Output("answer-box", "children"),
              [Input("button", "n_clicks")],
              [State(component_id="input-box", component_property="value")])
def render_answer_box(n_clicks, text):
    print(type(bow))
    print(type(clf))
    print(text)
    try:
        text_vec = bow.transform([text]).toarray()
        label = clf.predict(text_vec)
        if text_vec.sum()==0:
            label = "No Label"
    except:
        label = "No Label"
    return html.H2(label)


if __name__=="__main__":
    app.run_server(debug=True)