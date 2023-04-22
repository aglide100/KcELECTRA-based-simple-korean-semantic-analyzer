const onnx = require("onnxruntime-node");
const { BertTokenizer } = require("bert-tokenizer");

// const { AutoTokenizer } = require("@xenova/transformers");
// import { AutoTokenizer } from "@xenova/transformers";
// import * as onnx from "onnxruntime-node";
// import { InferenceSession } from "onnxruntime-node";
// import { AutoTokenizer } from "web-transformers";
// ONNX 모델 경로 설정
const modelPath = "classifier.onnx";

const MODEL_NAME = "beomi/KcELECTRA-base-v2022";

const vocabUrl =
    "https://huggingface.co/beomi/KcELECTRA-base-v2022/raw/main/vocab.txt";
console.log("!");

// 토크나이저 구현필요....
const test = async () => {
    try {
        console.log("!");
        const session = await onnx.InferenceSession.create(modelPath);
        const text = "한국어 텍스트";

        const bertTokenizer = new BertTokenizer(vocabUrl, true);
        console.log(bertTokenizer.tokenize(text));
        //[ 1045, 2066, 13137, 20968 ]
        // console.log(bertTokenizer.convertIdsToTokens(tokenIds));
        // //[ '▁i', '▁like', '▁straw', 'berries' ]
        // console.log(bertTokenizer.convertSingleExample(text));
        //[ '[CLS]', '▁i', '▁like', '▁straw', 'berries', '[SEP]' ]

        // const tokenizer = await AutoTokenizer.fromPretrained(MODEL_NAME);

        // const encoded = await create_model_input(question, context);
        // console.log("encoded", encoded);

        // const length = encoded.input_ids.length;
        // var input_ids = new Array(length);
        // var attention_mask = new Array(length);
        // var token_type_ids = new Array(length);

        // // Get encoded.input_ids as BigInt
        // input_ids[0] = BigInt(101);
        // attention_mask[0] = BigInt(1);
        // token_type_ids[0] = BigInt(0);
        // var i = 0;
        // for (; i < length; i++) {
        //     input_ids[i + 1] = BigInt(encoded.input_ids[i]);
        //     attention_mask[i + 1] = BigInt(1);
        //     token_type_ids[i + 1] = BigInt(0);
        // }
        // input_ids[i + 1] = BigInt(102);
        // attention_mask[i + 1] = BigInt(1);
        // token_type_ids[i + 1] = BigInt(0);

        // console.log("arrays", input_ids, attention_mask, token_type_ids);

        // const inputs = await tokenizer.encode(text);
        // console.log(inputs);
        // const inputIds = inputs[0].inputIds;
        // const attentionMask = inputs[0].attentionMask;
        // const tokenTypeIds = inputs[0].typeIds;

        // const output = await session.run({
        //     input_ids: inputIds,
        //     attention_mask: attentionMask,
        //     token_type_ids: tokenTypeIds,
        // });
        // console.log(output);
    } catch (e) {
        console.log("!---------");
        console.log(e);
    }
};

test();
