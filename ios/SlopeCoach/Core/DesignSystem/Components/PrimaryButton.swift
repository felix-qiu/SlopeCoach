import SwiftUI

struct PrimaryButton: View {
    let title: String
    var systemImage: String? = nil
    var isEnabled = true
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                if let systemImage {
                    Image(systemName: systemImage)
                        .font(.system(size: 17, weight: .semibold))
                }

                Text(title)
                    .font(.system(size: 16, weight: .semibold))
            }
            .frame(maxWidth: .infinity)
            .frame(height: 52)
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
        .foregroundStyle(isEnabled ? Color.white : Color(.systemGray2))
        .background(
            isEnabled ? Color.slopePrimary : Color(.systemGray5),
            in: RoundedRectangle(cornerRadius: 14, style: .continuous)
        )
        .shadow(
            color: isEnabled ? Color.slopePrimary.opacity(0.18) : .clear,
            radius: 8,
            y: 4
        )
        .accessibilityIdentifier("primary-button-\(title)")
    }
}

