// NeuroShield - iOS App with WatchKit integration for nightmare detection
// AppDelegate.swift

import UIKit
import CoreML
import WatchConnectivity

@main
class AppDelegate: UIResponder, UIApplicationDelegate {
    var session: WCSession?
    
    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        // Setup WatchConnectivity if available
        if WCSession.isSupported() {
            session = WCSession.default
            session?.delegate = self
            session?.activate()
        }
        return true
    }

    // MARK: UISceneSession Lifecycle
    func application(_ application: UIApplication, configurationForConnecting connectingSceneSession: UISceneSession, options: UIScene.ConnectionOptions) -> UISceneConfiguration {
        return UISceneConfiguration(name: "Default Configuration", sessionRole: connectingSceneSession.role)
    }
}

// WCSessionDelegate implementation
extension AppDelegate: WCSessionDelegate {
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        print("WCSession activation completed: \(activationState.rawValue), error: \(error?.localizedDescription ?? "none")")
        
        // Notify that watch connectivity state has changed
        NotificationCenter.default.post(name: .watchConnectivityChanged, object: nil)
    }
    
    func sessionDidBecomeInactive(_ session: WCSession) {
        print("WCSession became inactive")
        NotificationCenter.default.post(name: .watchConnectivityChanged, object: nil)
    }
    
    func sessionDidDeactivate(_ session: WCSession) {
        print("WCSession deactivated")
        // Reactivate session if needed
        WCSession.default.activate()
        NotificationCenter.default.post(name: .watchConnectivityChanged, object: nil)
    }
    
    func session(_ session: WCSession, didReceiveMessageData messageData: Data) {
        // Process the full data package from Apple Watch
        do {
            // Decode the JSON data package
            if let dataPackage = try JSONSerialization.jsonObject(with: messageData, options: []) as? [String: Any] {
                // Process the data package and extract relevant information
                processDataPackage(dataPackage)
            }
        } catch {
            print("Error decoding message data: \(error.localizedDescription)")
        }
    }
    
    private func processDataPackage(_ dataPackage: [String: Any]) {
        // Extract user ID
        if let userId = dataPackage["userId"] as? String {
            NightmareDetectionManager.shared.setUserId(userId)
        }
        
        // Process heart rate data
        if let heartRateData = dataPackage["heartRateData"] as? [[String: Any]] {
            for hrData in heartRateData {
                if let time = hrData["time"] as? Double,
                   let bpm = hrData["bpm"] as? Double {
                    // Notify about heart rate update for UI
                    NotificationCenter.default.post(
                        name: .heartRateUpdated,
                        object: nil,
                        userInfo: ["heartRate": bpm, "timestamp": time]
                    )
                    
                    // Process in nightmare detection manager
                    NightmareDetectionManager.shared.processHeartRateData(bpm, timestamp: time)
                }
            }
        }
        
        // Process motion data
        if let motionData = dataPackage["motionData"] as? [[String: Any]] {
            for motion in motionData {
                if let time = motion["time"] as? Double,
                   let x = motion["acceleration_x"] as? Double,
                   let y = motion["acceleration_y"] as? Double,
                   let z = motion["acceleration_z"] as? Double {
                    // Process in nightmare detection manager
                    NightmareDetectionManager.shared.processMotionData(x, y, z, timestamp: time)
                }
            }
        }
        
        // Process REM sleep information
        if let remInfo = dataPackage["remInfo"] as? [String: Any],
           let isREM = remInfo["isREM"] as? Bool {
            // Update REM sleep state in nightmare detection manager
            NightmareDetectionManager.shared.updateREMSleepState(isREM)
            
            // Notify UI about REM sleep state change
            NotificationCenter.default.post(
                name: .remSleepStateChanged,
                object: nil,
                userInfo: ["isREM": isREM]
            )
            
            // If REM sleep started, record the start time
            if isREM, let remStartTime = remInfo["remStartTime"] as? Double {
                NightmareDetectionManager.shared.setREMStartTime(remStartTime)
            }
        }
    }
}
