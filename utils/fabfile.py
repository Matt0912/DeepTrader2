from fabric.api import env
from fabric.api import run
from fabric.api import prompt
from fabric.api import execute
from fabric.api import sudo
from fabric.operations import run, put, get, settings
import time
import boto.ec2
import boto3

# add an environmental setting
env.aws_region = 'eu-west-2'

# add a function
def get_ec2_connection():
    if 'ec2' not in env:
        conn = boto.ec2.connect_to_region(env.aws_region)
        if conn is not None:
            env.ec2 = conn
            print("Connected to EC2 region %s" % env.aws_region)
        else:
            msg = "Unable to connect to EC2 region %s"
            raise IOError(msg % env.aws_region)
    return env.ec2

def list_aws_instances(verbose=False, state='all'):
    conn = get_ec2_connection()

    reservations = conn.get_all_reservations()
    instances = []
    for res in reservations:
        for instance in res.instances:
            if state == 'all' or instance.state == state:
                instance = {
                    'id': instance.id,
                    'type': instance.instance_type,
                    'image': instance.image_id,
                    'state': instance.state,
                    'instance': instance,
                }
                instances.append(instance)
    env.instances = instances
    if verbose:
        import pprint
        pprint.pprint(env.instances)


def select_instance(state='running'):
    #if env.active_instance:
     #   return

    list_aws_instances(state=state)

    

    prompt_text = "Please select from the following instances:\n"
    instance_template = " %(ct)d: %(state)s instance %(id)s\n"
    for idx, instance in enumerate(env.instances):
        ct = idx + 1
        args = {'ct': ct}
        args.update(instance)
        prompt_text += instance_template % args
    prompt_text += "Choose an instance: "

    def validation(input):
        choice = int(input)
        if not choice in range(1, len(env.instances) + 1):
            raise ValueError("%d is not a valid instance" % choice)
        return choice

    choice = prompt(prompt_text, validate=validation)
    env.active_instance = env.instances[choice - 1]['instance']

def getAllInstances(state='running'):

    list_aws_instances(state=state)
    list_of_instances = []

    for instance in env.instances:
        list_of_instances.append(instance['instance'])

    return list_of_instances
    

def run_command_on_selected_server(command):
    #select_instance()
    all_instances = getAllInstances()
    selected_hosts = []
    for instance in all_instances:
        selected_hosts.append('ec2-user@' + instance.public_dns_name)

    execute(command, hosts=selected_hosts)

def run_BSE_scripts(command):
    all_instances = getAllInstances()
    selected_hosts = []
    for instance in all_instances:
        selected_hosts.append('ec2-user@' + instance.public_dns_name)

    for i in range(0, len(selected_hosts)):
        execute(command, i, host=selected_hosts[i])

def setup_script():
    ## Install libraries needed
    sudo('yum install -y python-pip')
    sudo('pip install more_itertools')
    sudo('yum install -y tmux')
    put(r"C:\Users\Matt\Documents\CompSciLinux\Thesis\BristolStockExchange\BSE_ExpVM.py", r"/home/ec2-user")

def run_script(num):
    # Move over BSE.py so that it has latest version
    #sudo('scp -i ~/Desktop/random.pem ~/Desktop/hello_aws.py ec2-user@ec2-34-201-49-170.compute-1.amazonaws.com:/home/ec2-user')
    run("tmux new -d -s new_session")
    runBSE_command = ('python\ BSE_ExpVM.py\ ' + str(num))
    full_command = ("tmux send -t new_session.0 " + runBSE_command + " ENTER")
    run(full_command)

def retrieve_data_script():
    # IF BSE HAS FINISHED RUNNING (output = run("ps -A"))
    get(r"/home/ec2-user/*.csv", r"C:\Users\Matt\Documents\CompSciLinux\Thesis\BristolStockExchange\experiments\data_v4")

def check_if_running_script(num):
    output = run("ps -ef | grep 'python'")
    print("Instance " + str(num) + ":")
    print(output)
    



### CALL FUNCTIONS BELOW USING: fab -i ec2-keypair-ssh.openssh <function_name>:<function_variable>

def create_new_instances(num):
    ec2 = boto3.resource('ec2')

    # create a new EC2 instance
    instances = ec2.create_instances(
        ImageId='ami-04122be15033aa7ec',
        MinCount=1,
        MaxCount=int(num),
        InstanceType='t2.micro',
        SecurityGroupIds = ['sg-074b481ef501670d4'],
        KeyName='ec2-keypair'
    )

    print("Created " + str(num) + " new instances")

def initial_setup():
    run_command_on_selected_server(setup_script)

def run_BSE():
    run_BSE_scripts(run_script)

def retrieve_data():
    run_command_on_selected_server(retrieve_data_script)

def running_instances():
    run_BSE_scripts(check_if_running_script)


#scp -i "ec2-keypair-ssh.openssh" -r "ec2-18-133-74-97.eu-west-2.compute.amazonaws.com:/home/ec2-user" "C:\Users\Matt\Documents\CompSciLinux\Thesis\BristolStockExchange\experiments\data1"