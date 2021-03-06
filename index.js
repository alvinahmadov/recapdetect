import * as path  from 'path';
import {dirname}  from './src/utils.js'
import {Detector} from './src/detector.js';

const images = ['./data/samples/sample1.png', './data/samples/sample2.png'];

const weightsFile = path.join(dirname, './data/yolo/yolov3.weights');
const configFile = path.join(dirname, './data/yolo/yolov3.cfg');
const namesFile = path.join(dirname, './data/classes/coco.names');

(async () => {
	try {
		const detector = new Detector(weightsFile, configFile, namesFile, false);
		images.forEach((imagePath, i) => {
			detector.detect(imagePath, `./data/samples/predicted${i + 1}`);
		});
		
	} catch (e) {
		console.error(e);
	}
})()
