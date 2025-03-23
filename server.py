import os
import concurrent
import threading
import time
import grpc
import steg_service_pb2 as pb
import steg_service_pb2_grpc as pbgrpc
from concurrent import futures
from server_context import ServerContext
from services.auto import AutoService
from services.structural import StructualService
from services.calibration import CalibrationService
from services.feaext import FeaextService
from services.ml import MlService
# import multiprocessing

from pyutils import util
      
class AletheiaService():
    def __init__(self, server_context, max_workers=10):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.max_timeout = 5

        self.server_context = server_context
        self.auto_service = AutoService(self.server_context) 
        self.structual_service = StructualService()
        self.calibration_service = CalibrationService()       
        self.feaext_service = FeaextService()
        self.ml_service = MlService(self.server_context)

        # service definition
        self.steg_service_info = pb.StegServiceInfo()
        self.steg_service_info.name = "aletheia"
        self.steg_service_info.description = "Aletheia automated tools as a DFIR Steg-Hub Service"
        # service function definitions
        # auto
        func_auto = pb.StegServiceFunction(name="auto", description="Tries different steganalysis methods.")
        func_auto.return_fields.append(pb.StegServiceReturnFieldDefinition(name="outguess_pred", label="Prediction Outguess (JPG)", type=pb.Type.FLOAT, description="Propability Outguess"))
        func_auto.return_fields.append(pb.StegServiceReturnFieldDefinition(name="steghide_pred", label="Prediction Steghide (JPG)", type=pb.Type.FLOAT, description="Propability Steghide"))
        func_auto.return_fields.append(pb.StegServiceReturnFieldDefinition(name="nsf5_pred", label="Prediction nsF5 (JPG)", type=pb.Type.FLOAT, description="Propability nsF5"))
        func_auto.return_fields.append(pb.StegServiceReturnFieldDefinition(name="juniward_pred", label="Prediction J-Uniward (JPG)", type=pb.Type.FLOAT, description="Propability J-Uniward"))
        func_auto.return_fields.append(pb.StegServiceReturnFieldDefinition(name="lsbr_pred", label="Prediction LSBR (PNG)", type=pb.Type.FLOAT, description="Estimated Payload LSBR"))
        func_auto.return_fields.append(pb.StegServiceReturnFieldDefinition(name="lsbm_pred", label="Prediction LSBM (PNG)", type=pb.Type.FLOAT, description="Propability LSBM"))
        func_auto.return_fields.append(pb.StegServiceReturnFieldDefinition(name="steganogan_pred", label="Prediction SteganoGAN (PNG)", type=pb.Type.FLOAT, description="Propability SteganoGAN"))
        func_auto.return_fields.append(pb.StegServiceReturnFieldDefinition(name="uniward_pred", label="Prediction UNIWARD (PNG)", type=pb.Type.FLOAT, description="Propability UNIWARD"))
        func_auto.supported_file_types.append("png")
        func_auto.supported_file_types.append("jpg")
        self.steg_service_info.functions.append(func_auto)

        # effnetb0-predict
        func_effnetb0_predict = pb.StegServiceFunction(name="effnetb0_predict", description="Runs prediction on a pre-trained effnetb0 model.")
        func_effnetb0_predict.return_fields.append(pb.StegServiceReturnFieldDefinition(name="pred", label="Prediction result.", type=pb.Type.FLOAT, description="Propability of image containing stego payload"))
        func_effnetb0_predict.parameter.append(pb.StegServiceParameterDefinition(name="model_name", description="PNG:\nA-alaska2-hill\nA-alaska2-hilluniw\nA-alaska2-lsbm\nA-alaska2-lsbr\nA-alaska2-steganogan\nJPG:\nA-alaska2-f5\nA-alaska2-jmipod\nA-alaska2-juniw+wiener\nA-alaska2-juniw\nA-alaska2-nsf5\nA-alaska2-outguess\nA-alaska2-steghide\nA-alaska2-uniw", optional=False, type=pb.Type.STRING))
        func_effnetb0_predict.supported_file_types.append("png")
        func_effnetb0_predict.supported_file_types.append("jpg")
        self.steg_service_info.functions.append(func_effnetb0_predict)

    def Execute(self, request : pb.StegServiceRequest, context): 
        print(f"Received request from {context.peer()} for function {request.function}")

        timeout = request.request_timeout_sec if request.request_timeout_sec != 0 else self.max_timeout
        result_container = {"response": None}
        stop_event = threading.Event() 

        def task():
            try:
                result_container["response"] = self._execute(request, context)
            except Exception as e:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
                result_container["response"] = pb.StegServiceResponse()
            finally:
                stop_event.set()

        task_thread = threading.Thread(target=task, daemon=True)
        task_thread.start()

        start_time = time.time()

        while time.time() - start_time < timeout:
            if not context.is_active():
                print("Client disconnected before completion.")
                stop_event.set()  
                return pb.StegServiceResponse()

            if result_container["response"] is not None:
                return result_container["response"]

            time.sleep(0.01)

        context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
        context.set_details(f"Request timeout exceeded after {timeout} seconds")
        stop_event.set()  
        return pb.StegServiceResponse()
    
    def _execute(self, request : pb.StegServiceRequest, context):
        print(f"recieved request from {context.peer()}")
        
        try:
            match request.function:
                case "auto":
                    return util.parse_dict(self.auto_service.auto_method(request.file))
                case "effnetb0_predict":
                    return util.parse_dict(self.ml_service.effnetb0_predict(request.file, util.get_parameter(self.steg_service_info, "model_name", request)))
                case _:
                    return pb.StegServiceResponse(error="given function not supported")
        except Exception as ex:
            return pb.StegServiceResponse(error=str(ex))

    def GetStegServiceInfo(self, request, context):
        print(f"recieved GetStegServiceInfo request from {context.peer()}")
        return self.steg_service_info
        
def serve():
    # load server settings
    port = os.environ['port']
    if ":" not in port:
        port = f'[::]:{port}'
    max_workers = os.environ.get("max_workers")
    if max_workers == None:
        max_workers = 20
    models_path = os.environ.get("models_path")
    load_models_lazy = os.environ.get("load_models_lazy", "True").lower() != "false"

    server_context = ServerContext(models_path=models_path, load_models_lazy=load_models_lazy)
    
    MAX_MESSAGE_SIZE = 40 * 1024 * 1024
    
    aletheia_svc = AletheiaService(server_context, max_workers)
    grpc_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    
    server = grpc.server(
        grpc_executor,                         
        options=[
        ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),  
        ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE) 
        ],)

    pbgrpc.add_StegServiceServicer_to_server(aletheia_svc, server)
    server.add_insecure_port(port)
    print(f"{aletheia_svc.steg_service_info.name} service started on port {port}")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
