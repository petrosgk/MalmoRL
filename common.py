def parse_clients_args(args_clients):
    """
    Return an array of tuples (ip, port) extracted from ip:port string
    :param args_clients:
    :return:
    """
    return [str.split(str(client), ':') for client in args_clients]
