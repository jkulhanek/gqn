import pytorch_lightning as pl
import inspect
import logging
import sys
try:
    from typing import Literal
except(ImportError):
    # Literal polyfill
    class _Literal:
        @classmethod
        def __getitem__(cls, key):
            tp = key[0] if isinstance(key, tuple) else key
            return type(tp)
    Literal = _Literal()


def setup_logging(level=logging.INFO):
    from tqdm import tqdm

    def is_console_handler(handler):
        return isinstance(handler, logging.StreamHandler) and handler.stream in {sys.stdout, sys.stderr}

    class TqdmLoggingHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:  # noqa pylint: disable=bare-except
                self.handleError(record)

    logging.basicConfig(stream=sys.stdout, level=level)
    handler = TqdmLoggingHandler(sys.stdout)
    try:
        import colorlog
        formatter = colorlog.LevelFormatter(fmt={
            'DEBUG': '%(log_color)sdebug: %(message)s (%(module)s:%(lineno)d)%(reset)s',
            'INFO': '%(log_color)sinfo%(reset)s: %(message)s',
            'WARNING': '%(log_color)swarning%(reset)s: %(message)s (%(module)s:%(lineno)d)',
            'ERROR': '%(log_color)serror%(reset)s: %(message)s (%(module)s:%(lineno)d)',
            'CRITICAL': '%(log_color)scritical: %(message)s (%(module)s:%(lineno)d)%(reset)s',
        }, log_colors={
            'DEBUG': 'white',
            'INFO': 'bold_green',
            'WARNING': 'bold_yellow',
            'ERROR': 'bold_red',
            'CRITICAL': 'bold_red',
        })
        handler.setFormatter(formatter)
    except(ModuleNotFoundError):
        # We do not require colorlog to be present
        pass
    logging._acquireLock()
    orig_handlers = logging.root.handlers
    try:
        logging.root.handlers = [x for x in orig_handlers if not is_console_handler(x)] + [handler]
    except Exception:
        logging.root.handlers = orig_handlers
    finally:
        logging._releaseLock()


def get_parameters(function_or_cls, output_all=False):
    if inspect.isclass(function_or_cls):
        def collect_parameters(cls):
            output = []
            parameters = inspect.signature(cls.__init__).parameters
            for p in parameters.values():
                if p.kind == inspect.Parameter.VAR_KEYWORD:
                    for base in cls.__bases__:
                        output.extend(collect_parameters(base))
                if p.kind == inspect.Parameter.KEYWORD_ONLY or p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    output.append(p)
            return output

        params = collect_parameters(function_or_cls)
    else:
        params = []
        parameters = inspect.signature(function_or_cls).parameters
        for p in parameters.values():
            if p.kind == inspect.Parameter.KEYWORD_ONLY or p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                params.append(p)
    output_parameters = []
    for p in params:
        output_parameters.append(p)
    return output_parameters


def add_arguments(parser, function_or_cls):
    params = get_parameters(function_or_cls)
    for p in params:
        if p.default is None:
            continue
        if p.annotation is None:
            continue
        name = p.name.replace('_', '-')
        annotation = p.annotation
        help = ''
        if getattr(getattr(annotation, '__origin__', None), '_name', None) == 'Literal':
            parser.add_argument(f'--{name}', type=type(annotation.__args__[0]), choices=annotation.__args__, default=p.default, help=f'{help} [{p.default}]')
        elif isinstance(p.default, bool):
            parser.set_defaults(**{p.name: p.default})
            parser.add_argument(f'--{name}', dest=p.name, action='store_true', help=f'{help} [{p.default}]')
            parser.add_argument(f'--no-{name}', dest=p.name, action='store_false', help=f'{help} [{p.default}]')
        elif annotation in [int, float, str]:

            parser.add_argument(f'--{name}', type=annotation, default=p.default, help=f'{help} [{p.default}]')
    return parser


def bind_arguments(args, function_or_cls):
    kwargs = {}
    args_dict = args.__dict__
    parameters = get_parameters(function_or_cls)
    for p in parameters:
        if p.name in args_dict:
            kwargs[p.name] = args_dict[p.name]
    return kwargs
