from jinja2 import Environment, FileSystemLoader

file_loader = FileSystemLoader('templates')
env = Environment(loader=file_loader)

template = env.get_template('model_template.txt')
ouput = ''
def render_template(template_dict):
    global output
    
    output = template.render(template_dict=template_dict)
    
    with open('model/'+template_dict['class_name']+'.py', 'w') as f:
        f.write(output)


def get_ouput():
    global output
    return output